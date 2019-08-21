#! /usr/bin/env python
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: David R Thompson, david.r.thompson@jpl.nasa.gov
#

import os
import sys
import json
import argparse
import scipy as s
from spectral.io import envi
from scipy.io import savemat
from common import load_config, expand_all_paths, load_spectrum
from forward import ForwardModel
from inverse_mpi import Inversion
from inverse_mcmc import MCMCInversion
from geometry import Geometry
from fileio_mpi import IO
import cProfile
import logging
from mpi4py import MPI
from mpiutils import Tags, send_io, recv_io

# Suppress warnings that don't come from us
import warnings
warnings.filterwarnings("ignore")

def main():

    description = 'Spectroscopic Surface & Atmosphere Fitting'
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--level', default='INFO')
    parser.add_argument('--row_column', default='')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(format='%(message)s', level=args.level)

    # Load the configuration file.
    config = load_config(args.config_file)

    # Build the forward model and inversion objects.
    fm = ForwardModel(config['forward_model'])
    if 'mcmc_inversion' in config:
        iv = MCMCInversion(config['mcmc_inversion'], fm)
    else:
        iv = Inversion(config['inversion'], fm)

    # We set the row and column range of our analysis. The user can
    # specify: a single number, in which case it is interpreted as a row;
    # a comma-separated pair, in which case it is interpreted as a
    # row/column tuple (i.e. a single spectrum); or a comma-separated
    # quartet, in which case it is interpreted as a row, column range in the
    # order (line_start, line_end, sample_start, sample_end) - all values are
    # inclusive. If none of the above, we will analyze the whole cube.
    rows, cols = None, None

    if len(args.row_column) > 0:
        ranges = args.row_column.split(',')

        if len(ranges) == 1:
            rows, cols = [int(ranges[0])], None

        if len(ranges) == 2:
            row_start, row_end = ranges
            rows, cols = range(int(row_start), int(row_end)), None

        elif len(ranges) == 4:
            row_start, row_end, col_start, col_end = ranges
            line_start, line_end, samp_start, samp_end = ranges
            rows = range(int(row_start), int(row_end))
            cols = range(int(col_start), int(col_end))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()

    tags = Tags([
        'READY', # worker -> head, ready to receive IO or EXIT
        'PREPARE', # head -> worker, signal to call recv_io
        'WRITE', # worker -> head, to_write info from inversion
        'ROW', # head -> worker, row number
        'COL', # head -> worker, col number
        'MEAS', # head -> worker, None or size and data of measurement
        'GEOM', # head -> worker, obs then glt then loc
        'CONF', # head -> worker, configs tuple
        'EXIT' # head -> worker, no more jobs to do
    ])

    if rank == 0:
        print("### MPI size: %i" % size)
        active = {n : None for n in range(1, size)}
        io = IO(config, fm, iv, rows, cols)

        if 'radiometry_correction_file' in io.outfiles:
            logging.warning('radiometric calibration not supported with MPI!')
            logging.warning('all radiometric information is ignored')

        complete = False

        closed = 0
        while closed < size - 1:
            data = comm.recv(source=MPI.ANY_SOURCE, status=status)
            worker = status.Get_source()
            tag = status.Get_tag()

            if tag == tags.READY:
                try:
                    row, col, meas, geom, configs = next(io)
                    if meas is not None and all(meas < -49.0):
                        print('READY tag - Row Col: %i %i' % (row, col))
                        #print('col: %s' % str(col))
                        #io.write_spectrum(row, col, [], meas, geom)
                        #io.write_spectrum(row, col, dict())
                        states = []
                        #continue

                except StopIteration:
                    complete = True

                if not complete:
                    comm.send(None, dest=worker, tag=tags.PREPARE)
                    send_io(row, col, meas, geom, configs, comm, worker, tags)
                    active[worker] = (row, col)
                else:
                    comm.send(None, dest=worker, tag=tags.EXIT)
                    closed = closed + 1

            elif tag == tags.WRITE:
                row, col = active[worker]
                to_write = data

                io.write_spectrum(row, col, to_write)
                active[worker] = None

        # WO: Maybe flush_buffers here after the while loop

    else:
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            data = comm.recv(source=0, status=status)
            tag = status.Get_tag()

            if tag == tags.PREPARE:
                row, col, meas, geom, configs = recv_io(comm, tags, status)
                print("#### worker rank %i doing row %i and col %i" % (rank,row,col))

                # No guarantee that profiling will still work with MPI.
                if args.profile:
                    gbl, lcl = globals(), locals()
                    cProfile.runctx('iv.invert(meas, geom, configs)', gbl, lcl)

                else:
                    to_write = iv.invert(row, col, meas, geom, configs)
                    comm.send(to_write, dest=0, tag=tags.WRITE)

            elif tag == tags.EXIT:
                break

if __name__ == '__main__':
    main()
