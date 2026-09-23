#%%
import random
import numpy as np

def random_graph(n, p = 0.25):

    nodes = np.arange(n)
    edges = []
    weights = np.ones((1,n))

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                edges.append([i, j])

    return nodes, edges, weights


def read_graph(filename):
    nodes = []
    edges = []

    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split()

            # Leere Zeilen ignorieren
            if not parts:
                continue

            # Problemzeile: p edge n m
            if parts[0] == "p":
                n = int(parts[2])
                nodes = list(range(n))

            # Kante: e x y
            elif parts[0] == "e":
                x = int(parts[1])
                y = int(parts[2])

                edges.append([x-1, y-1])

    return nodes, edges



import timeit

def benchmark(objekt, methoden, repeat=5, number=10):

    ergebnisse = {}

    for methode, args in methoden:

        funktion = getattr(objekt, methode)
        zeiten = timeit.repeat(
            lambda: funktion(*args),
            repeat=repeat,
            number=number
        )

        zeiten = [zeit / number for zeit in zeiten]

        ergebnisse[args] = {
            "min": min(zeiten)
            #"max": max(zeiten),
            #"mittelwert": sum(zeiten) / len(zeiten)
        }

    return ergebnisse


#%% random_tests
if __name__ == "__main__":

    testdateien = [5,10,15,20,25,30]

    wkeiten = [0.25,0.4,0.55,0.7]

    methoden = [
            ("maxclique_solve", ("Lovasz",)),
            ("maxcut_solve_GW", ("GW",)),
            ("maxcut_solve_Nt", ("Nt",)),
            ("maxclique_solve", ("Old",)),
            ("maxclique_solve", ("Ostergard",)),
            ("maxclique_solve", ("Randomized",))
            
        ]
    testergebnisse = {}

    from max_clique_solve_lovasz import MaxCliqueSolve

    for wkeit in wkeiten:
        for datei in testdateien:
            knoten, kanten, gewichte = random_graph(datei,wkeit)

            dateiname = f"graph_n{datei}_p{wkeit}.txt"

            with open(dateiname, "w", encoding="utf-8") as datei:
                datei.write("Knoten:\n")
                datei.write(str(knoten) + "\n\n")

                datei.write("Kanten:\n")
                datei.write(str(kanten) + "\n\n")

                datei.write("Gewichte:\n")
                datei.write(str(gewichte) + "\n")

            maxcut = MaxCliqueSolve(knoten, gewichte, kanten)

            zeiten = benchmark(maxcut, methoden)
                        
            testergebnisse[dateiname] = {"Zeiten": zeiten, 
                                         "          Old": maxcut.maxclique_solve('Old'), 
                                         "Ostergard": maxcut.maxclique_solve('Ostergard'),
                                         "Randomized ": maxcut.maxclique_solve('Randomized'),
                                         "Lovasz ": maxcut.maxclique_solve('Lovasz'),
                                         "GW ": maxcut.maxcut_solve_GW(),
                                         "Nt ": maxcut.maxcut_solve_Nt()}
    with open("testergebnisse.txt", "w", encoding="utf-8") as datei:
        for schluessel, wert in testergebnisse.items():
            datei.write(f"{schluessel}: {wert}\n")   

               