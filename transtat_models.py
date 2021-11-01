"""
This module defines a network-based epidemic model.

Classes: EpiModel
"""

import heapq as hp
import csv

import numpy as np
import scipy.stats as stat
import networkx as nx


class EpiModel:
    """
    Base class for stochastic SEIR epidemic models. The pairwise contact
    interval distribution is exponential(1) by default. In the pairwise and
    external transmission models, covariates act according to an accelerated
    failure time model, acting multiplicatively on the rate parameter.

    Methods
        epidemic: Runs an epidemic to completion or until a specified number
            of infections is reached

    """
    def __init__(self,
                 digraph, CIdist=stat.expon(), xCIdist=None,
                 pnames=None, pcoef=None, xnames=None, xcoef=None):
        """Initialize network-based stochastic SEIR model.

        Arguments
            digraph: NetworkX DiGraph with pairwise covariates for internal
                transmission stored as digraph[i][j]["xij"] for edge ij.
                Latent periods, infectious periods, and individual-level
                covariates for node i are stored in digraph.nodes[i] under the
                keys "latpd", "infpd", and "xi", respectively. Any node labels
                except None can be used.
            CIdist: SciPy.stats distribution for contact interval when all
                covariates equal zero. Default is exponential(1).
            xCIdist: SciPy.stats distribution for external contact interval
                when all covariates equal zero.
            pnames: A tuple of names for each covariate in the internal
                transmission model.
            pcoef:  Coefficient tuple matching pairwise covariates.
                Coefficients are log rate ratios.
            xnames: A tuple of names for each covariate in the external
                transmission model.
            xcoef: Coefficient tuple matching individual-level covariates.
                Coefficients are log rate ratios as in pcoef.

        """
        # directed graph with covariate data; remove self-loops.
        if nx.number_of_selfloops(digraph) > 0:
            raise Exception("Self-loops in digraph not allowed.")
        self.n = digraph.number_of_nodes()
        self.digraph = digraph

        # internal and external contact interval distributions
        self.CIdist = CIdist
        self.xCIdist = xCIdist

        # pairwise covariates and coefficients
        if pnames is not None:
            if pcoef is not None and len(pnames) != len(pcoef):
                raise Exception(
                    "Pairwise names and coefficients have different lengths."
                )
            elif pcoef is None:
                raise Exception("Pairwise coefficient values expected.")
        elif pcoef is not None:
            raise Exception("Pairwise covariate names expected.")
        self.pnames = pnames
        self.pcoef = pcoef

        # external covariates and coefficients
        if xnames is not None:
            if xcoef is not None and len(xnames) != len(xcoef):
                raise Exception(
                    "External names and coefficients have different lengths."
                )
            elif xcoef is None:
                raise Exception("External coefficient values expected.")
        elif xcoef is not None:
            raise Exception("External covariate names expected.")
        self.xnames = xnames
        self.xcoef = xcoef

        # epidemic data
        self.pdata = None
        self.xdata = None

    def external_contacts(self):
        """Generate contact intervals from external sources.

        Arguments
            atrisk: List of nodes at risk of external infectious
                contact.

        """
        nullCIlist = self.xCIdist.rvs(size=self.n)
        if self.xcoef is None:
            CIlist = nullCIlist
        else:
            lnrates = np.array([
                np.dot(self.xcoef, self.digraph.nodes[i]["xi"])
                for i in self.digraph.nodes()
            ])
            CIlist = nullCIlist / np.exp(lnrates)
        return(zip(self.digraph.nodes(), CIlist))

    def infectious_contacts(self, i, t, neighbors):
        """Generate contact intervals from i to susceptible neighbors.

        Arguments
            i: Node index.
            t: Infection time of i.
            neighbors: List of susceptible neighbors (successor nodes)
                of i.

        Returns
            Dictionary with three keys: 'neighbors' is a list of
            susceptible neighbors (successor nodes) of i, 'contacts'
            contains (tij, j) for each j with whom i makes infectious
            contact and 'escapes' contains (t + latpd + infpd, j) for
            all other j at risk of infectious contact from i.

        """
        neighbors = np.array(neighbors)
        latpd = self.digraph.nodes[i]["latpd"]
        infpd = self.digraph.nodes[i]["infpd"]

        # generate contact intervals
        nullCIlist = self.CIdist.rvs(size=len(neighbors))
        if self.pcoef is None:
            CIlist = nullCIlist
        else:
            idict = self.digraph[i]
            lnrates = [
                np.dot(self.pcoef, idict[j]["xij"]) for j in neighbors
            ]
            CIlist = nullCIlist / np.exp(lnrates)

        # test whether contact interval <= infpd
        CItest = np.less_equal(CIlist, infpd)

        # return dictionary of contacts and escapes
        return {
            'neighbors': neighbors,
            'contacts': zip(t + latpd + CIlist[CItest], neighbors[CItest]),
            'escapes': zip(
                [t + latpd + infpd] * sum(~CItest), neighbors[~CItest]
            )
        }


    def epidemic(self, stop_size=None, external=None, attempts=100):
        """Run an epidemic to completion or to a number of infections.

        Arguments
            stop_size: The number of infections after which observation
                of the epidemic will be stopped. If None, will run to
                completion.
            external: List containing (node, external contact time) for
                each possible imported infection. If not specified,
                external contacts will be generated using the
                external_contacts method.
            attempts: The maximum number of attempts that will be made
                to obtain at least "stop_size" infections. Set to one
                if stop_size = None.

        Returns (adds attributes to self)
            self.pdata: List of (i, j, entry, exit, infector, infset)
                + covariates where infector indicates whether i
                infected j, and infset indicates whether i is in the
                infectious set of j.
            self.xdata: List of (j, entry, exit, infector, infset)
                + covariates, where infector indicates whether j was
                infected from an external source, and infset indicates
                whether the infectious set of j contains an external
                source.

        """
        size = att = 0
        if stop_size is None:
            stop_size = self.n
            attempts = 1
        while size < stop_size and att < attempts:
            # initialize data dictionaries
            Etime = dict()      # Infection times
            Itime = dict()      # Onset of infectiousness times
            Rtime = dict()      # End of infectiousness times

            # initialize data lists
            pdata = []      # pairwise transmission data
            xdata = []      # external transmission data
            escapes = []    # (i, j) where j escapes infectious contact from i
            xinf = []       # individuals infected from external sources

            # run epidemic
            size = 0
            epiheap = []
            hp.heapify(epiheap)

            if external is None:
                external = self.external_contacts()
            for (i, t) in external:
                # push (infectious contact time, i, source) onto heap
                hp.heappush(epiheap, (t, i, None))
            while epiheap and size < stop_size:
                t, i, vi = hp.heappop(epiheap)
                if i not in Etime:
                    # i infected at time ti = t
                    size += 1

                    # get latent and infectious periods
                    latpd = self.digraph.nodes[i]["latpd"]
                    infpd = self.digraph.nodes[i]["infpd"]
                    Etime[i] = t
                    Itime[i] = t + latpd
                    Rtime[i] = t + latpd + infpd

                    # record data for i
                    # pdata is (vi, i, Itime[vi], t, infector, infset)
                    if vi is not None:
                        # i infected internally by vi
                        pdata.append((vi, i, Itime[vi], t, 1, 1))
                    else:
                        xinf.append(i)

                    # infectious contacts from i to susceptible neighbors
                    neighbors = [
                        j for j in self.digraph.successors(i)
                        if j not in Etime
                    ]
                    inf_contacts = self.infectious_contacts(i, t, neighbors)

                    # update heap and escapes
                    for (tij, j) in inf_contacts['contacts']:
                        # j susceptible at time ti; push tij onto heap
                        hp.heappush(epiheap, (tij, j, i))
                    for (tij, j) in inf_contacts['escapes']:
                        # time at risk of transmission to be determined
                        escapes.append((i, j))

                elif vi is not None and Itime[vi] < Etime[i]:
                    pdata.append((vi, i, Itime[vi], Etime[i], 0, 1))

            # define end of observation
            if stop_size == self.n:
                # observation until removal of last infected
                T = t + latpd + infpd
            else:
                # observation until stop_size^th infection time
                T = t

            # record remaining data from epiheap
            # pdata is (i, j, Itime[vi], exit, infector, infset)
            while epiheap:
                tij, j, i = hp.heappop(epiheap)
                if j not in Etime:
                    # j not infected
                    if i is not None and Itime[i] < T:
                        # j exposed to i and i still infectious (tij > T)
                        pdata.append((i, j, Itime[i], T, 0, 0))
                else:
                    # j infected
                    if i is not None and Itime[i] < Etime[j]:
                        # j exposed to i and i infectious at tj
                        pdata.append((i, j, Itime[i], Etime[j], 0, 1))

            # record data from escapes
            for (i, j) in escapes:
                if Itime[i] < T:
                    # i infectious before end of observation at T
                    if j not in Etime:
                        # j not infected
                        exitt = min(T, Rtime[i])
                        infset = 0
                    else:
                        # j infected by source other than i
                        if Etime[j] > Itime[i] and Etime[j] < Rtime[i]:
                            # i is in the infectious set of j
                            exitt = Etime[j]
                            infset = 1
                        else:
                            exitt = Rtime[i]
                            infset = 0
                    pdata.append((i, j, Itime[i], exitt, 0, infset))

        if pdata:
            # record pairwise transmission data
            if self.pcoef is not None:
                # add pairwise covariates
                pdata = [
                    prow + self.digraph[i][j]["xij"] for prow in pdata
                ]
            self.pdata = pdata

        if xinf:
            # record external transmission data
            # xdata is (i, entry, exit, infector, infset) + covariates
            for i in xinf:
                # i infected from an external source
                xdata.append((i, 0, Etime[i], 1, 1))
            for i in set(self.digraph.nodes()).difference(xinf):
                if i in Etime:
                    exitt = Etime[i]
                    infset = 1
                else:
                    exitt = T
                    infset = 0
                xdata.append((i, 0, exitt, 0, infset))
            if self.xcoef is not None:
                # add individual-level covariates
                xdata = [
                    xrow + self.digraph.nodes[i]["xi"] for xrow in xdata
                ]
            self.xdata = xdata

    def write_pdata(self, filename):
        """Write pairwise transmission data to csv file.

        Arguments
            filename: Name of csv file.

        """
        pvars = (
            "infectious", "susceptible", "entry", "exitt", "infector", "infset"
        )
        if self.pnames:
            pvars += self.pnames
        with open(filename, "w", encoding = "utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(pvars)
            for row in self.pdata:
                writer.writerow(row)
            csvfile.close()

    def write_xdata(self, filename):
        """Write external transmission data to csv file.

        Arguments
            filename: Name of csv file.

        """
        xvars = (
            "susceptible", "entry", "exit", "infector", "infset", "trace"
        )
        if self.xnames:
            xvars += self.xnames
        with open(filename, "w", encoding = "utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(xvars)
            for row in self.xdata:
                writer.writerow(row)
            csvfile.close()
