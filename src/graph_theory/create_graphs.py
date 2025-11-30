import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
coordinates = {
    1: (0, 0),
    2: (1, 2),
    3: (30, 1),
    4: (14, 33),
    5: (2, 12),
    6: (20, 4)
}

for node, coord in coordinates.items():
    G.add_node(node, pos=coord)

for u in G.nodes():
    for v in G.nodes():
        if u != v:
            dist = ((coordinates[u][0] - coordinates[v][0])**2 + (coordinates[u][1] - coordinates[v][1])**2)**0.5
            G.add_edge(u, v, weight=dist)

edges_with_weights = G.edges(data=True)

for edge in edges_with_weights:
    u, v, data = edge
    weight = data.get('weight') 
    print(f"Kante ({u}, {v}) - Gewicht: {weight}")

# plot graph
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=500, font_size=10, font_color='black')
plt.show()

# intersections
def check_intersection(edge1, edge2, pos):
    x1, y1 = pos[edge1[0]]
    x2, y2 = pos[edge1[1]]
    x3, y3 = pos[edge2[0]]
    x4, y4 = pos[edge2[1]]

    if (edge1[0] == edge2[0] or edge1[0] == edge2[1] or
        edge1[1] == edge2[0] or edge1[1] == edge2[1]):
        return False  

    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if det == 0:  # edges parallel 
        return False

    intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

    if (min(x1, x2) <= intersection_x <= max(x1, x2) and
            min(y1, y2) <= intersection_y <= max(y1, y2) and
            min(x3, x4) <= intersection_x <= max(x3, x4) and
            min(y3, y4) <= intersection_y <= max(y3, y4)):
        return True

    return False

# check intersections
for edge1 in G.edges():
    for edge2 in G.edges():
        if edge1 != edge2 and edge1[::-1] != edge2: 
            if check_intersection(edge1, edge2, pos):
                print(f"Edges {edge1} & {edge2} are intersecting.")