import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Demandez à l'utilisateur de saisir le nombre de nœuds (doit être au moins 4)
num_nodes = int(input("Entrez le nombre de nœuds : "))

# Vérifiez que le nombre de nœuds est au moins 4
if num_nodes < 4:
    print("Le nombre de nœuds doit être au moins 4.")
else:
    # Créez un graphe en forme d'anneau (cycle)
    G = nx.cycle_graph(num_nodes)

    # Créez un maillage partiel aléatoire
    maillage_partiel = nx.random_geometric_graph(num_nodes, 0.3)

    # Ajoutez les arêtes du maillage partiel au graphe en forme d'anneau
    G.add_edges_from(maillage_partiel.edges())

    # Extrait la centralité de degré de chaque nœud
    degree_centrality = nx.degree_centrality(G)
    degree_values = np.array([degree_centrality[node] for node in G.nodes()]).reshape(-1, 1)

    # Appliquez K-Means (en trois clusters)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(degree_values)
    cluster_assignments = kmeans.labels_

    # Créez des sous-graphes pour chaque cluster
    subgraphs = [G.subgraph([node for node, cluster in enumerate(cluster_assignments) if cluster == i]) for i in range(3)]

    # Créez un dictionnaire de couleurs pour chaque cluster
    colors = ['r', 'g', 'b']  # Vous pouvez personnaliser les couleurs

    # Affichez chaque sous-graphe avec des couleurs en fonction des clusters
    for i, subgraph in enumerate(subgraphs):
        pos = nx.spring_layout(subgraph)
        color = [colors[i]] * len(subgraph.nodes)
        plt.figure(i)
        nx.draw(subgraph, pos, with_labels=True, node_size=300, node_color=color)
        plt.title(f"Cluster {i+1} - Graphe avec {len(subgraph.nodes)} nœuds")
        plt.show()
