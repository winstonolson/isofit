#! /usr/bin/env python3
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

import sys
import scipy as s
from common import svd_inv, svd_inv_sqrt, eps, resample_spectrum
from collections import OrderedDict
from scipy.optimize import least_squares
import xxhash
from inverse_simple import invert_simple, invert_algebraic
from scipy.linalg import inv, norm, det, cholesky, qr, svd
from scipy.linalg import LinAlgError
from hashlib import md5
from numba import jit
import time
import logging

error_code = -1


class Inversion:

    def __init__(self, config, forward):

        # Initialization specifies retrieval subwindows for calculating
        # measurement cost distributions
        self.lasttime = time.time()
        self.fm = forward
        self.method = 'GradientDescent'
        self.hashtable = OrderedDict()  # Hash table for caching inverse matrices
        self.max_table_size = 500
        self.windows = config['windows']  # Retrieval windows
        self.state_indep_S_hat = False
        self.windows = config['windows']
        self.simulation_mode = None
        if 'simulation_mode' in config:
            self.simulation_mode = config['simulation_mode']
        if 'Cressie_MAP_confidence' in config:
            self.state_indep_S_hat = config['Cressie_MAP_confidence']

        # We calculate the instrument channel indices associated with the
        # retrieval windows using the initial instrument calibration.  These
        # window indices never change throughout the life of the object.
        self.winidx = s.array((), dtype=int)  # indices of retrieval windows
        for lo, hi in self.windows:
            idx = s.where(s.logical_and(self.fm.instrument.wl_init > lo,
                                        self.fm.instrument.wl_init < hi))[0]
            self.winidx = s.concatenate((self.winidx, idx), axis=0)
        self.counts = 0
        self.inversions = 0

        # Finally, configure Levenberg Marquardt.
        self.least_squares_params = {'method': 'trf', 'max_nfev': 20,
                                     'bounds': (self.fm.bounds[0]+eps, self.fm.bounds[1]-eps),
                                     'x_scale': self.fm.scale, 'xtol': 1e-4, 'ftol': 1e-4, 'gtol': 1e-4,
                                     'tr_solver': 'exact'}
        for k, v in config.items():
            if k in self.least_squares_params:
                self.least_squares_params[k] = v

    @jit
    def calc_prior(self, x, geom):
        """Calculate prior distribution of radiance.  This depends on the
        location in the state space.  Return the inverse covariance and
        its square root (for non-quadratic error residual calculation)"""

        xa = self.fm.xa(x, geom)
        Sa = self.fm.Sa(x, geom)
        Sa_inv, Sa_inv_sqrt = svd_inv_sqrt(Sa, hashtable=self.hashtable)
        return xa, Sa, Sa_inv, Sa_inv_sqrt

    @jit
    def calc_posterior(self, x, geom, meas):
        """Calculate posterior distribution of state vector. This depends
        both on the location in the state space and the radiance (via noise)."""

        xa = self.fm.xa(x, geom)
        Sa = self.fm.Sa(x, geom)
        Sa_inv = svd_inv(Sa, hashtable=self.hashtable)
        K = self.fm.K(x, geom)
        Seps = self.fm.Seps(x, meas, geom)
        Seps_inv = svd_inv(Seps, hashtable=self.hashtable)

        # Gain matrix G reflects current state, so we use the state-dependent
        # Jacobian matrix K
        S_hat = svd_inv(K.T.dot(Seps_inv).dot(K) + Sa_inv,
                        hashtable=self.hashtable)
        G = S_hat.dot(K.T).dot(Seps_inv)

        # N. Cressie [ASA 2018] suggests an alternate definition of S_hat for
        # more statistically-consistent posterior confidence estimation
        if self.state_indep_S_hat:
            Ka = self.fm.K(xa, geom)
            S_hat = svd_inv(Ka.T.dot(Seps_inv).dot(Ka) + Sa_inv,
                            hashtable=self.hashtable)
        return S_hat, K, G

    @jit
    def calc_Seps(self, x, meas, geom):
        """Calculate (zero-mean) measurement distribution in radiance terms.
        This depends on the location in the state space. This distribution is
        calculated over one or more subwindows of the spectrum. Return the
        inverse covariance and its square root."""

        Seps = self.fm.Seps(x, meas, geom)
        wn = len(self.winidx)
        Seps_win = s.zeros((wn, wn))
        for i in range(wn):
            Seps_win[i, :] = Seps[self.winidx[i], self.winidx]
        return svd_inv_sqrt(Seps_win, hashtable=self.hashtable)

    def invert(self, row, col, meas, geom, configs):
        """Inverts a meaurement and returns a state vector."""

        self.lasttime = time.time()
        self.trajectory = []
        self.counts = 0

        # Simulations are easy - return the initial state vector
        if self.simulation_mode or meas is None:
            states = s.array([self.fm.init.copy()])

        else:

            # Calculate the initial solution, if needed.
            x0 = invert_simple(self.fm, meas, geom)

            # Seps is the covariance of "observation noise" including both
            # measurement noise from the instrument as well as variability due
            # to unknown variables.  For speed, we will calculate it just once
            # based on the initial solution (a potential minor source of
            # inaccuracy)
            Seps_inv, Seps_inv_sqrt = self.calc_Seps(x0, meas, geom)

            @jit
            def jac(x):
                """Calculate measurement jacobian and prior jacobians with
                respect to COST function.  This is the derivative of cost with
                respect to the state.  The Cost is expressed as a vector of
                'residuals' with respect to the prior and measurement,
                expressed in absolute terms (not quadratic) for the solver,
                It is the square root of the Rodgers et. al Chi square version.
                All measurement distributions are calculated over subwindows
                of the full spectrum."""

                # jacobian of measurment cost term WRT state vector.
                K = self.fm.K(x, geom)[self.winidx, :]
                meas_jac = Seps_inv_sqrt.dot(K)

                # jacobian of prior cost term with respect to state vector.
                xa, Sa, Sa_inv, Sa_inv_sqrt = self.calc_prior(x, geom)
                prior_jac = Sa_inv_sqrt

                # The total cost vector (as presented to the solver) is the
                # concatenation of the "residuals" due to the measurement
                # and prior distributions. They will be squared internally by
                # the solver.
                total_jac = s.concatenate((meas_jac, prior_jac), axis=0)

                return s.real(total_jac)

            def err(x):
                """Calculate cost function expressed here in absolute terms
                (not quadratic) for the solver, i.e. the square root of the
                Rodgers et. al Chi square version.  We concatenate 'residuals'
                due to measurment and prior terms, suitably scaled.
                All measurement distributions are calculated over subwindows
                of the full spectrum."""

                # Measurement cost term.  Will calculate reflectance and Ls from
                # the state vector.
                est_meas = self.fm.calc_meas(x, geom, rfl=None, Ls=None)
                est_meas_window = est_meas[self.winidx]
                meas_window = meas[self.winidx]
                meas_resid = (est_meas_window-meas_window).dot(Seps_inv_sqrt)

                # Prior cost term
                xa, Sa, Sa_inv, Sa_inv_sqrt = self.calc_prior(x, geom)
                prior_resid = (x - xa).dot(Sa_inv_sqrt)

                # Total cost
                total_resid = s.concatenate((meas_resid, prior_resid))

                # How long since the last call?
                newtime = time.time()
                secs = newtime-self.lasttime
                self.lasttime = newtime

                self.trajectory.append(x)

                it = len(self.trajectory)
                tm = newtime - self.lasttime
                rs = sum(pow(total_resid, 2))
                sm = self.fm.summarize(x, geom)
                logging.debug('Iteration: %02i  Residual: %12.2f %s' %
                              (it, rs, sm))

                return s.real(total_resid)

            # Initialize and invert
            try:
                xopt = least_squares(err, x0, jac=jac,
                    **self.least_squares_params)
                self.trajectory.append(xopt.x)
            except LinAlgError:
                logging.warning('Levenberg Marquardt failed to converge')

            states = s.array(self.trajectory)

        self.fm.reconfigure(*configs)

        if len(states) == 0:

            # Write a bad data flag
            atm_bad = s.zeros(len(self.fm.statevec)) * -9999.0
            state_bad = s.zeros(len(self.fm.statevec)) * -9999.0
            data_bad = s.zeros(self.fm.instrument.n_chan) * -9999.0
            to_write = {'estimated_state_file': state_bad,
                        'estimated_reflectance_file': data_bad,
                        'estimated_emission_file': data_bad,
                        'modeled_radiance_file': data_bad,
                        'apparent_reflectance_file': data_bad,
                        'path_radiance_file': data_bad,
                        'simulated_measurement_file': data_bad,
                        'algebraic_inverse_file': data_bad,
                        'atmospheric_coefficients_file': atm_bad,
                        'spectral_calibration_file': data_bad,
                        'posterior_uncertainty_file': state_bad}

        else:
            # The inversion returns a list of states, which are
            # intepreted either as samples from the posterior (MCMC case)
            # or as a gradient descent trajectory (standard case). For
            # gradient descent the last spectrum is the converged solution.
            if self.method == 'MCMC':
                state_est = states.mean(axis=0)
            else:
                state_est = states[-1, :]

            # Spectral calibration
            wl, fwhm = self.fm.calibration(state_est)
            cal = s.column_stack(
                [s.arange(0, len(wl)), wl / 1000.0, fwhm / 1000.0])

            # If there is no actual measurement, we use the simulated version
            # in subsequent calculations.  Naturally in these cases we're
            # mostly just interested in the simulation result.
            if meas is None:
                meas = self.fm.calc_rdn(state_est, geom)

            # Rodgers diagnostics
            lamb_est, meas_est, path_est, S_hat, K, G = \
                self.forward_uncertainty(state_est, meas, geom)

            # Simulation with noise
            meas_sim = self.fm.instrument.simulate_measurement(meas_est, geom)

            # Algebraic inverse and atmospheric optical coefficients
            x_surface, x_RT, x_instrument = self.fm.unpack(state_est)
            rfl_alg_opt, Ls, coeffs = invert_algebraic(self.fm.surface,
                self.fm.RT, self.fm.instrument, x_surface, x_RT, x_instrument,
                meas, geom)
            rhoatm, sphalb, transm, solar_irr, coszen, transup = coeffs
            atm = s.column_stack(list(coeffs[:4]) +
                [s.ones((len(wl), 1)) * coszen])

            # Upward emission & glint and apparent reflectance
            Ls_est = self.fm.calc_Ls(state_est, geom)
            apparent_rfl_est = lamb_est + Ls_est

            # Assemble all output products
            to_write = {'estimated_state_file': state_est,
                        'estimated_reflectance_file':
                        s.column_stack((self.fm.surface.wl, lamb_est)),
                        'estimated_emission_file':
                        s.column_stack((self.fm.surface.wl, Ls_est)),
                        'estimated_reflectance_file':
                        s.column_stack((self.fm.surface.wl, lamb_est)),
                        'modeled_radiance_file':
                        s.column_stack((wl, meas_est)),
                        'apparent_reflectance_file':
                        s.column_stack((self.fm.surface.wl, apparent_rfl_est)),
                        'path_radiance_file':
                        s.column_stack((wl, path_est)),
                        'simulated_measurement_file':
                        s.column_stack((wl, meas_sim)),
                        'algebraic_inverse_file':
                        s.column_stack((self.fm.surface.wl, rfl_alg_opt)),
                        'atmospheric_coefficients_file':
                        atm,
                        'spectral_calibration_file':
                        cal,
                        'posterior_uncertainty_file':
                        s.sqrt(s.diag(S_hat))}

            return to_write

    def forward_uncertainty(self, x, meas, geom):
        """Converged estimates of path radiance, radiance, reflectance
        # Also calculate the posterior distribution and Rodgers K, G matrices"""

        dark_surface = s.zeros(self.fm.surface.wl.shape)
        path = self.fm.calc_meas(x, geom, rfl=dark_surface)
        mdl = self.fm.calc_meas(x, geom, rfl=None, Ls=None)
        lamb = self.fm.calc_lamb(x, geom)
        S_hat, K, G = self.calc_posterior(x, geom, meas)
        return lamb, mdl, path, S_hat, K, G
