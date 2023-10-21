import networkx as nx
import matplotlib.pyplot as plt

# Demandez à l'utilisateur de saisir le nombre de nœuds
num_nodes = int(input("Enter the number of nodes (must be at least 4) : "))

# Vérifiez que le nombre de nœuds est au moins 4
if num_nodes < 4:
    print("The number of nodes must be at least 4.")
else:
    # Créez un graphe en forme d'anneau (cycle)
    G = nx.cycle_graph(num_nodes)

    # Créez un maillage partiel aléatoire
    maillage_partiel = nx.random_geometric_graph(num_nodes, 0.3)

    # Ajoutez les arêtes du maillage partiel au graphe en forme d'anneau
    G.add_edges_from(maillage_partiel.edges())

    # Affichez le graphe
    pos = nx.spring_layout(G)  
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue')
    plt.title(f"Graphe avec {num_nodes} nœuds en forme d'anneau et maillage partiel")
    plt.show()
