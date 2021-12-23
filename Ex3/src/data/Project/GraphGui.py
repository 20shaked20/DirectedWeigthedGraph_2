import sys

from DiGraph import DiGraph
from GraphAlgo import GraphAlgo
import pygame
import math

SIZE = 1000, 1000
RED = (255, 0, 0)
BLACK = (0, 0, 0)
PINK = (255, 200, 200)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

"""
LEGACY CLASS, not using!!! moved to work on gui via matplotlib.
"""


class GraphGui:

    def __init__(self, graph_algo: GraphAlgo):
        self.graph_algo = graph_algo

        pygame.init()
        self.screen = pygame.display.set_mode(SIZE)
        self.screen.fill(WHITE)
        self.nodesX = {}
        self.nodesY = {}
        self.update_x_y_pos()
        self.draw_line_arrows()
        self.draw_nodes()
        self.run()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # Flip the display < this has to be done idk why.
            pygame.display.flip()

    def draw_nodes(self):
        """
        This method initializes the ui.
        i'e -> creates objects ( lines, elevators, floors... )
        """
        pygame.draw.rect(self.screen, (0, 0, 255), (0, 0, 1000, 0), 30)  # pygame clicker box.
        nodes = self.graph_algo.get_graph().get_all_v()
        font = pygame.font.Font('freesansbold.ttf', 10)
        for k in nodes:
            curr_x = self.nodesX[k]
            curr_y = self.nodesY[k]
            curr_id = k
            update_pos = (curr_x, curr_y)
            pygame.draw.circle(self.screen, BLUE, (update_pos[0], update_pos[1]), 10)
            id_pos = font.render(str(curr_id), True, WHITE, BLUE)
            rect_pos = id_pos.get_rect()
            rect_pos.center = (update_pos[0], update_pos[1])
            self.screen.blit(id_pos, rect_pos)

    def draw_line_arrows(self):
        nodes = self.graph_algo.get_graph().get_all_v()
        for k in nodes:
            out_edge = self.graph_algo.get_graph().all_out_edges_of_node(k)
            start = [self.nodesX[k], self.nodesY[k]]
            for v in out_edge:
                end = [self.nodesX[v], self.nodesY[v]]
                self.draw_arrow(BLACK, start, end)

    def update_x_y_pos(self):
        minX = sys.float_info.max
        minY = sys.float_info.max
        maxX = sys.float_info.min
        maxY = sys.float_info.min

        nodes = self.graph_algo.get_graph().get_all_v()

        for k in nodes:
            curr_pos = nodes[k]
            x = float(curr_pos[0])
            y = float(curr_pos[1])
            minX = min(minX, x)
            minY = min(minY, y)
            maxX = max(maxX, x)
            maxY = max(maxY, y)

        X_Scaling = 1000 / (maxX - minX) * 0.9
        Y_Scaling = 1000 / (maxY - minY) * 0.9

        for k in nodes:
            curr_id = k
            curr_pos = nodes[k]
            x = (float(curr_pos[0]) - minX) * X_Scaling + 10
            y = (float(curr_pos[1]) - minY) * Y_Scaling + 30

            self.nodesX[curr_id] = int(x)
            self.nodesY[curr_id] = int(y)

    def draw_arrow(self, colour, start, end):  # start = [x,y] end = [x,y]
        pygame.draw.line(self.screen, colour, start, end, 2)
        rotation = math.degrees(math.atan2(start[1] - end[1], end[0] - start[0])) + 90
        pygame.draw.polygon(self.screen, BLACK, (
            (end[0] - 5 * math.sin(math.radians(rotation)), end[1] - 5 * math.cos(math.radians(rotation))),
            (
                end[0] + 30 * math.sin(math.radians(rotation - 120)),
                end[1] + 30 * math.cos(math.radians(rotation - 120))),
            (end[0] + 30 * math.sin(math.radians(rotation + 120)),
             end[1] + 30 * math.cos(math.radians(rotation + 120)))))


if __name__ == '__main__':
    nodes = {}
    graph = DiGraph(nodes)
    graph_algo = GraphAlgo(graph)
    graph_algo.load_from_json("/Users/Shaked/PycharmProjects/DirectedWeigthedGraph_2/Ex3/data/A0.json")
    run = GraphGui(graph_algo)
