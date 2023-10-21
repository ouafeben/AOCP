import matplotlib.pyplot as plt
import networkx as nx
from numpy import array

G = nx.Graph()
# definition des noeuds
#G.add_node(0,label='0',col='pink')
G.add_node(1,label='1',col='red')
G.add_node(2,label='2',col='green')
G.add_node(3,label='3',col='red')
G.add_node(4,label='4',col='red')
G.add_node(5,label='5',col='blue')
G.add_node(6,label='6',col='green')
G.add_node(7,label='7',col='blue')
G.add_node(8,label='8',col='green')
G.add_node(9,label='9',col='green')
G.add_node(10,label='10',col='green')
G.add_node(11,label='11',col='blue')
G.add_node(12,label='12',col='blue')
G.add_node(13,label='13',col='green')
G.add_node(14,label='14',col='green')
G.add_node(15,label='15',col='blue')
G.add_node(16,label='16',col='green')
G.add_node(17,label='17',col='green')
G.add_node(18,label='18',col='green')
G.add_node(19,label='19',col='blue')
G.add_node(20,label='20',col='red')
G.add_node(21,label='21',col='red')
G.add_node(22,label='22',col='blue')
G.add_node(23,label='23',col='red')
G.add_node(24,label='24',col='red')
# definition des aretes
G.add_edge(1,2,styl='dashed')
G.add_edge(1,4,styl='solid')
G.add_edge(1,5,weight=1,styl='solid')
G.add_edge(2,3,weight=3,styl='solid')
G.add_edge(2,5,weight=8,styl='solid')
G.add_edge(2,9,weight=8,styl='solid')
G.add_edge(2,6,weight=8,styl='solid')
G.add_edge(3,6,weight=6,styl='dashed')
G.add_edge(3,7,weight=6,styl='dashed')
G.add_edge(4,8,weight=9,styl='solid')
G.add_edge(4,12,weight=9,styl='solid')
G.add_edge(5,9,weight=9,styl='solid')
G.add_edge(5,8,weight=9,styl='solid')
G.add_edge(6,7,weight=9,styl='solid')
G.add_edge(6,9,weight=9,styl='solid')
G.add_edge(6,10,weight=9,styl='solid')
G.add_edge(7,10,weight=9,styl='solid')
G.add_edge(7,11,weight=9,styl='solid')
G.add_edge(8,9,weight=9,styl='solid')
G.add_edge(8,12,weight=9,styl='solid')
G.add_edge(8,13,weight=9,styl='solid')
G.add_edge(9,13,weight=9,styl='solid')
G.add_edge(9,14,weight=9,styl='solid')
G.add_edge(10,11,weight=9,styl='solid')
G.add_edge(10,14,weight=9,styl='solid')
G.add_edge(10,15,weight=9,styl='solid')
G.add_edge(11,15,weight=9,styl='solid')
G.add_edge(11,24,weight=9,styl='solid')
G.add_edge(12,16,weight=9,styl='solid')
G.add_edge(12,19,weight=9,styl='solid')
G.add_edge(13,14,weight=9,styl='solid')
G.add_edge(13,16,weight=9,styl='solid')
G.add_edge(13,17,weight=9,styl='solid')
G.add_edge(14,15,weight=9,styl='solid')
G.add_edge(14,17,weight=9,styl='solid')
G.add_edge(15,18,weight=9,styl='solid')
G.add_edge(16,17,weight=9,styl='solid')
G.add_edge(16,19,weight=9,styl='solid')
G.add_edge(16,20,weight=9,styl='solid')
G.add_edge(17,18,weight=9,styl='solid')
G.add_edge(17,20,weight=9,styl='solid')
G.add_edge(17,22,weight=9,styl='solid')
G.add_edge(18,22,weight=9,styl='solid')
G.add_edge(18,23,weight=9,styl='solid')
G.add_edge(18,24,weight=9,styl='solid')
G.add_edge(19,20,weight=9,styl='solid')
G.add_edge(19,21,weight=9,styl='solid')
G.add_edge(20,21,weight=9,styl='solid')
G.add_edge(20,22,weight=9,styl='solid')
G.add_edge(21,22,weight=9,styl='solid')
G.add_edge(22,23,weight=9,styl='solid')
G.add_edge(23,24,weight=9,styl='solid')
liste = list(G.nodes(data='col'))
colorNodes = {}
for noeud in liste:
    colorNodes[noeud[0]]=noeud[1]
colorNodes
colorList=[colorNodes[node] for node in colorNodes]
colorList
liste = list(G.nodes(data='label'))
labels_nodes = {}
for noeud in liste:
    labels_nodes[noeud[0]]=noeud[1]
labels_nodes
labels_edges = {}
#labels_edges = {edge:G.edges[edge]['weight'] for edge in G.edges}
#labels_edges = {edge:'' for edge in G.edges}
labels_edges
liste = list(G.edges(data='styl'))
edges_style = {}
edges_style = {edge:G.edges[edge]['styl'] for edge in G.edges}
edges_style
# positions for all nodes
pos = nx.spring_layout(G)  

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700,node_color=colorList,alpha=0.9)
               
# labels
nx.draw_networkx_labels(G, pos, labels=labels_nodes, \
                        font_weight=20, \
                        font_color='k', \
                        font_family='sans-serif')

# edges
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_edges, font_color='tab:red')


plt.axis('off')
plt.savefig(r'E:\data\fig1.png')
plt.show()
