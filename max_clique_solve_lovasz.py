#%%
import numpy as np
from typing import List
from numpy.typing import NDArray
import random
import cvxpy as cp
import math

class MaxCliqueSolve:
    def __init__(self, vertices: NDArray, weights: NDArray, edges: List[List]):
        self.vertices = vertices
        self.weights = weights
        self.edges = edges
        self.n = len(vertices)
    '''TODO: Schreibe Check, ob vertices ein 1d-Array in Form einer Range ist, kontrolliere, dass die weights gleichlang 
             wie die vertices sind und kontrolliere, dass es keine Duplikate/Permutationen oder "Paare" in den edges gibt.'''

    def clique_to_cut(self):
        '''TODO: check if output ist sinnvoll gewählt oder die Funktion noch erweitert werden muss um besseren Output liefern zu können'''
        w = 2 * (self.n - 1) * np.max(self.weights)
        W = np.ones((self.n, self.n)) * -w
        for edge in self.edges:
            W[edge[0], edge[1]] = self.weights[0,edge[0]] + self.weights[0,edge[1]]
            W[edge[1], edge[0]] = self.weights[0,edge[0]] + self.weights[0,edge[1]]
        for i in range(self.n):
            W[i,i] = 0        
        e = np.ones((self.n, 1))
        W_top = np.hstack((e.T @ W @ e, e.T @ W))
        W_bottom = np.hstack((W @ e, W))
        return 1/4 * np.vstack((W_top, W_bottom))


    def maxcut_solve_GW(self, arg = "GW"):
        """
        Goemans-Williamson-Algorithmus für das Max-Cut Problem.

        Arbeitet auf der verschobenen Gewichtsmatrix W_shift aus _shift_weights(),
        die aus clique_to_cut() gewonnen wird. Die Verschiebung stellt sicher,
        dass alle Einträge der unteren Dreiecksmatrix nicht-negativ sind.

        Vorgehen:
          1. Berechne W_dach = clique_to_cut(), verschiebe zu W_shift via _shift_weights()
          2. SDP:  max  sum_{i<j} W_shift_ij * (1 - Y_ij)
                   s.t. Y >= 0 (psd),  Y_ii = 1 für alle i
          3. Cholesky-Zerlegung von Y -> Vektoren v_i
          4. Zufälliger Hyperebenen-Schnitt: wähle r ~ N(0, I),
             setze x_i = sign(r^T v_i)
          5. Wiederhole Schritt 4 mehrfach, behalte besten Schnitt.
          6. Rückübersetzung: Knoten auf der dem Lift-Knoten gegenüberliegenden
             Seite sind Clique-Kandidaten.

        Returns
        -------
        cut_value         : float        – Gewicht des Schnitts (bzgl. W_dach, unverschoben)
        S                 : list of int  – Knoten auf Seite +1 (IDs 0..n-1, ohne Lift-Knoten)
        S_bar             : list of int  – Knoten auf Seite -1
        clique_candidates : list of int  – Clique-Kandidaten (Seite gegenüber Lift-Knoten)
        """
        W_dach = self.clique_to_cut()
        W_shift = W_dach
        n1 = W_shift.shape[0]   # = n + 1

        # SDP (Maximierungsproblem):
        #   max  sum_{i<j} W_shift_ij * (1 - Y_ij)
        #   s.t. Y >= 0,  Y_ii = 1
        Y = cp.Variable((n1, n1), symmetric=True)
        constraints = [Y >> 0] + [Y[i, i] == 1 for i in range(n1)]

        obj_expr = 0.5 * cp.sum(cp.multiply(W_shift, Y))
        prob = cp.Problem(cp.Maximize(obj_expr), constraints)
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-6)

        if prob.status not in ("optimal", "optimal_inaccurate") or Y.value is None:
            raise RuntimeError(f"SDP nicht lösbar: Status = {prob.status}")

        #Y ≈ V V^T  →  Vektoren v_i = V[i, :]
        Y_val = (Y.value + Y.value.T) / 2      # Symmetrisierung
        eigvals, eigvecs = np.linalg.eigh(Y_val)
        eigvals = np.maximum(eigvals, 0)        # PSD-Projektion
        V = eigvecs @ np.diag(np.sqrt(eigvals))

        # Hyperebenen-Rounding: mehrere Versuche, besten Schnitt behalten
        best_cut = -np.inf
        best_S, best_S_bar, best_clique = [], [], []

        for _ in range(max(1000, 10 * n1)):
            r = np.random.randn(n1)
            x = np.sign(V @ r)
            x[x == 0] = 1                   # Grenzfall
            if x[0] == -1:
                x = x * -1

            cut = 0.5 * x @ W_dach @ x

            if cut > best_cut:
                best_cut = cut
                best_S     = [i - 1 for i in range(1, n1) if x[i] ==  1]
                best_S_bar = [i - 1 for i in range(1, n1) if x[i] == -1]

        return best_cut, best_S, best_S_bar


    def maxcut_solve_Nt(self, arg = "Nt"):

        """Berechnet die Schranke für den Max-Cut-Wert nach Nesterov (1998):

            (2/pi) * s*(L) + (1 - 2/pi) * s_*(L)

        wobei L = W_dach aus clique_to_cut() und:

            s*(L)  = max  (1/2) L • X    s.t. X >= 0, diag(X) = e
            s_*(L) = min  (1/2) L • X    s.t. X >= 0, diag(X) = e

        Returns
        -------
        bound    : float  – Nesterov-Schranke
        s_upper  : float  – Optimalwert s*(L)
        s_lower  : float  – Optimalwert s_*(L)"""
        L = self.clique_to_cut()
        n1 = L.shape[0]

        e = np.ones(n1)

        # Gemeinsame Constraints für beide SDPs
        def make_constraints(X):
            return [X >> 0] + [X[i, i] == 1 for i in range(n1)]

        # s*(L): Maximierungsproblem
        X_max = cp.Variable((n1, n1), symmetric=True)
        prob_max = cp.Problem(
            cp.Maximize(0.5 * cp.trace(L @ X_max)),
            make_constraints(X_max)
        )
        prob_max.solve(solver=cp.SCS, verbose=False, eps=1e-6)

        if prob_max.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"s*(L) SDP nicht lösbar: Status = {prob_max.status}")
        s_upper = float(prob_max.value)

        # s_*(L): Minimierungsproblem
        X_min = cp.Variable((n1, n1), symmetric=True)
        prob_min = cp.Problem(
            cp.Minimize(0.5 * cp.trace(L @ X_min)),
            make_constraints(X_min)
        )
        prob_min.solve(solver=cp.SCS, verbose=False, eps=1e-6)

        if prob_min.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"s_*(L) SDP nicht lösbar: Status = {prob_min.status}")
        s_lower = float(prob_min.value)

        bound = (2 / np.pi) * s_upper + (1 - 2 / np.pi) * s_lower
        bound2 = (2 / np.pi) * s_upper

        return bound, bound2, s_upper, s_lower

    def maxclique_solve(self, arg='Old'):
        if arg == 'Old':
            self.m = 0
            self.c = []
            self.v = []
            self.clique_rec(self.vertices, 0)
            return self.m, sorted(self.c)

        elif arg == 'Ostergard':
            self.m = 0
            self.cs = [1]
            self.c = []
            for i in range(self.n-1, -1, -1):
                self.found = False
                vi = self.vertices[i]
                U = np.intersect1d(self.vertices[i:], self.N(vi))
                self.v = [int(vi)]
                self.clique_rec_o(U, 1)
                self.v = []
                self.cs.insert(0, self.m)
            return self.cs[0], sorted(self.c)

        elif arg == 'Randomized':
            self.m = 0
            self.c = []
            self.v = []
            self.clique_rec_r(self.vertices, 0)
            return self.m, self.c

        elif arg == 'Lovasz':
            self.m = 0
            self.c = []
            self.v = []
            self._q_sdp, self._Q_sdp = self._solve_sdp_dual(self.vertices)
            self._idx_map = {int(v): i for i, v in enumerate(self.vertices)}
            self._lovasz_lb_init()
            self._clique_rec_lovasz(self.vertices, 0)
            return self.m, sorted(self.c)

        else:
            raise NotImplementedError(f'Das Argument {arg} ist nicht erlaubt.')

    # -----------------------------------------------------------------------
    # Lovász Branch-and-Bound Hilfsfunktionen
    # -----------------------------------------------------------------------

    def _lovasz_lb_init(self):
        """
        Initiale untere Schranke für die Cliquenzahl via Kantendichte:

            LB = ceil(1 / (1 - delta)),   delta = 2m / n^2
        """
        delta = (2 * len(self.edges)) / (self.n ** 2)
        lb = self.n if delta >= 1.0 else math.ceil(1.0 / (1.0 - delta))
        if lb > self.m:
            self.m = lb

    def _complement_edges(self, U):
        """
        Kantenmenge des Komplementgraphen G_bar[U].
        Eine Clique in G entspricht einer stabilen Menge in G_bar.

        Returns  set of (int, int)  – Paare {u,v} mit u < v in G_bar[U]
        """
        edge_set = {(min(a, b), max(a, b)) for a, b in self.edges}
        U_list = [int(u) for u in U]
        return {
            (u, v)
            for i, u in enumerate(U_list)
            for v in U_list[i+1:]
            if (min(u, v), max(u, v)) not in edge_set
        }

    def _solve_sdp_dual(self, U):
        """
        Löst das duale SDP (SDP-D) auf dem Komplementgraphen G_bar[U]:

            min   t
            s.t.  Q_ii = -2*q_i - w_i   for all i in U
                  Q_ij = 0               for all {i,j} in E(G_bar)
                  [[t, q^T], [q, Q]] >= 0  (PSD)

        Einheitsgewichte w_i = 1. Gibt (q, Q) als numpy-Arrays zurück.
        """
        k = len(U)
        U_list = [int(u) for u in U]
        idx = {u: i for i, u in enumerate(U_list)}

        t = cp.Variable()
        q = cp.Variable(k)
        Q = cp.Variable((k, k), symmetric=True)

        M = cp.bmat([[cp.reshape(t, (1, 1)), cp.reshape(q, (1, k))],
                     [cp.reshape(q, (k, 1)), Q]])
        constraints = [M >> 0]
        for i in range(k):
            constraints.append(Q[i, i] == -2 * q[i] - 1)  # w_i = 1
        for (u, v) in self._complement_edges(U):
            constraints.append(Q[idx[u], idx[v]] == 0)

        prob = cp.Problem(cp.Minimize(t), constraints)
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-6)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            return np.zeros(k), np.zeros((k, k))
        return q.value, Q.value

    def _vsdp(self, S_local, q, Q):
        """
        Wertet V_SDP(S) = q_S^T Q_S^† q_S aus (Lemma 2.7 im Paper).

        Nutzt die QP-Charakterisierung: V_SDP(S) = -min_y  y^T Q_S y - 2 q_S^T y
        mit Regularisierung lambda = 1e-4 für numerische Stabilität.
        """
        if len(S_local) == 0:
            return 0.0
        q_S = q[S_local]
        Q_S = Q[np.ix_(S_local, S_local)] + 1e-4 * np.eye(len(S_local))
        try:
            val = float(q_S @ np.linalg.solve(Q_S, q_S))
        except np.linalg.LinAlgError:
            val = float(q_S @ np.linalg.pinv(Q_S) @ q_S)
        return max(val, 0.0)

    def _lovasz_rest(self, U):
        """
        Obere Schranke theta(G_bar[U]) für die Cliquenzahl auf G[U] via SDP:

            theta(G_bar) = max  J • X
                           s.t. X >= 0,  tr(X) = 1,  X_ij = 0  for all {i,j} in E(G_bar[U])
        """
        k = len(U)
        if k == 0:
            return 0.0
        if k == 1:
            return 1.0

        U_list = [int(u) for u in U]
        idx = {u: i for i, u in enumerate(U_list)}
        X = cp.Variable((k, k), symmetric=True)
        constraints = [X >> 0, cp.trace(X) == 1]
        for (u, v) in self._complement_edges(U):
            constraints.append(X[idx[u], idx[v]] == 0)

        prob = cp.Problem(cp.Maximize(cp.trace(np.ones((k, k)) @ X)), constraints)
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-5)

        if prob.status not in ("optimal", "optimal_inaccurate") or prob.value is None:
            return float(k)
        return float(prob.value)

    def _rounding(self, U, q, Q):
        """
        Rounding nach Algorithm 2 (Gong, Cifuentes, Toriello 2025).

        Konstruiert eine Clique S in G[U] via while-Schleife auf G_bar[U]:

            S <- {}, I <- U
            while I != {}:
                i* <- isolierter Knoten / Blatt in G_bar[I]
                      ODER argmax_j [ V_SDP(I ohne ({j} und delta_j)) + w_j ]
                S <- S und {i*}
                I <- I ohne ({i*} und delta_{i*}^G_bar)

        Gibt die konstruierte Clique S zurück.
        """
        S = []
        I = [int(u) for u in U]

        while len(I) > 0:
            comp_neighbors = {u: set() for u in I}
            for (u, v) in self._complement_edges(np.array(I)):
                comp_neighbors[u].add(v)
                comp_neighbors[v].add(u)

            # Bevorzuge isolierte Knoten oder Blätter in G_bar[I]
            chosen = next((u for u in I if len(comp_neighbors[u]) <= 1), None)

            # Sonst: argmax V_SDP(I \ ({j} u delta_j)) + w_j
            if chosen is None:
                chosen = max(
                    I,
                    key=lambda u: self._vsdp(
                        [self._idx_map[v] for v in I
                         if v not in (comp_neighbors[u] | {u}) and v in self._idx_map],
                        q, Q
                    ) + 1.0
                )

            S.append(chosen)
            I = [v for v in I if v not in (comp_neighbors[chosen] | {chosen})]

        return S

    def _clique_rec_lovasz(self, U, size):
        """
        Branch-and-Bound mit Lovász-Schranke.

        Bound:    size + theta(G_bar[U]) <= LB(G[U])  ->  prunen
                  LB(G[U]) = ceil(1 / (1 - delta_U))  (Kantendichte-Schranke)
        Branching: zufälliger Knoten aus der via Algorithm 2 konstruierten Clique.
        """
    
        if len(U) == 0:
            if size > self.m:
                self.m = size
                self.c = self.v.copy()
            return

        while len(U) != 0:
            U_set = set(int(u) for u in U)
            m_U = sum(1 for (a, b) in self.edges if a in U_set and b in U_set)
            delta_U = (2 * m_U) / (len(U) ** 2)
            lb_U = len(U) if delta_U >= 1.0 else math.ceil(1.0 / (1.0 - delta_U))

            if size + self._lovasz_rest(U) <= lb_U:
                return

            vi = random.choice(self._rounding(U, self._q_sdp, self._Q_sdp))

            self.v.append(vi)
            self._clique_rec_lovasz(np.intersect1d(U, self.N(vi)), size + 1)
            self.v.pop()

            U = U[U != vi]
            

    # -----------------------------------------------------------------------
    # Bestehende Hilfsfunktionen
    # -----------------------------------------------------------------------

    def clique_rec(self, U, size):
        if len(U) == 0:
            if size > self.m:
                self.m = size
                self.c = self.v.copy()
            return
        while len(U) != 0:
            if size + len(U) <= self.m:
                return
            i = np.argmin(U)
            self.v.append(int(U[i]))
            Vi = U[i]
            U = np.delete(U, i)
            self.clique_rec(np.intersect1d(U, self.N(Vi)), size + 1)
            self.v.pop()

    def clique_rec_r(self, U, size):
        if len(U) == 0:
            if size > self.m:
                self.m = size
                self.c = self.v.copy()
            return
        while len(U) != 0:
            if size + len(U) <= self.m:
                return
            gewichte = [self.weights[:, i][0] for i in U]
            Wsum = sum(gewichte)
            wkeit = [x / Wsum for x in gewichte]
            i = np.random.choice(len(U), p=wkeit)
            self.v.append(int(U[i]))
            Vi = U[i]
            U = np.delete(U, i)
            self.clique_rec_r(np.intersect1d(U, self.N(Vi)), size + 1)
            self.v.pop()

    def clique_rec_o(self, U, size):
        '''TODO: Mitschreiben der Clique klappt nicht: FIXEN'''
        if len(U) == 0:
            if size > self.m:
                self.m = size
                self.c = self.v.copy()
                self.found = True
            return
        while len(U) != 0:
            if size + len(U) <= self.m:
                return
            i = np.argmin(U)
            u = int(U[i])
            if size + self.cs[i] <= self.m:
                return
            self.v.append(u)
            Vi = U[i]
            self.clique_rec_o(np.intersect1d(np.delete(U, i), self.N(Vi)), size + 1)
            self.v.pop()
            if self.found:
                return
            U = np.delete(U, i)

    def N(self, v):
        """Nachbarschaft von Knoten v als Array."""
        return np.array([b if a == v else a for a, b in self.edges if v in (a, b)])


