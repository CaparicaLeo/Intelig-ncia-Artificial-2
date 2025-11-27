import random
import math
import matplotlib.pyplot as plt

class AntColony:
    def __init__(self, distances, cities, n_ants, n_iterations, rho, alpha, beta, Q=100):
        self.distances  = distances
        self.cities     = cities
        # Inicializa feromônio (tau)
        self.pheromone  = [[1 / (len(distances) * len(distances)) for _ in range(len(distances))] for _ in range(len(distances))]
        self.n_ants     = n_ants
        self.n_iterations = n_iterations
        self.rho        = rho
        self.alpha      = alpha
        self.beta       = beta
        self.Q          = Q

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", float('inf'))
        
        # Plota o estado inicial (apenas cidades)
        print(">> Feche a janela do gráfico para iniciar a simulação...")
        self.plot_graph(title=f"Mapa Inicial ({len(self.cities)} Cidades)")

        print(f"\nIniciando simulação por {self.n_iterations} iterações...")

        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.update_pheromone(all_paths)
            
            # Acha o melhor desta iteração
            shortest_path = min(all_paths, key=lambda x: x[1])
            
            # Compara com o melhor global
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
                print(f"  [Nova melhor rota] Iteração {i+1}: Distância = {all_time_shortest_path[1]:.2f}")
            
        # Plota o resultado final
        print(">> Simulação finalizada. Gerando gráfico final...")
        self.plot_graph(path=all_time_shortest_path[0], 
                        title=f"Melhor Rota (Dist: {all_time_shortest_path[1]:.2f}) | Alpha={self.alpha}, Beta={self.beta}, Rho={self.rho}")
                
        return all_time_shortest_path

    def update_pheromone(self, all_paths):
        # 1. Evaporação: (1 - rho) * tau
        decay_factor = 1 - self.rho
        for i in range(len(self.pheromone)):
            for j in range(len(self.pheromone)):
                self.pheromone[i][j] *= decay_factor

        # 2. Depósito: Q / Distância_Total
        for path, dist in all_paths:
            if dist == 0: continue
            deposit_amount = self.Q / dist
            for move in path:
                self.pheromone[move[0]][move[1]] += deposit_amount
                self.pheromone[move[1]][move[0]] += deposit_amount

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele[0]][ele[1]]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(random.randint(0, len(self.cities)-1)) # Começa em cidade aleatória
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start)) # Fecha o ciclo
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = list(pheromone)
        dist = list(dist)
        
        # Zera prob de visitados
        for v in visited:
            pheromone[v] = 0
            
        row_calc = []
        for i in range(len(pheromone)):
            if i not in visited:
                heuristic = (1.0 / dist[i]) if dist[i] > 0 else 0
                val = (pheromone[i] ** self.alpha) * (heuristic ** self.beta)
                row_calc.append(val)
            else:
                row_calc.append(0)

        total_prob = sum(row_calc)
        
        if total_prob == 0:
            available = [i for i in range(len(pheromone)) if i not in visited]
            return random.choice(available)

        pick = random.uniform(0, total_prob)
        current = 0
        for i, val in enumerate(row_calc):
            current += val
            if current >= pick:
                return i
        return -1

    def plot_graph(self, path=None, title="Grafo"):
        x = [c[0] for c in self.cities]
        y = [c[1] for c in self.cities]

        plt.figure(figsize=(10, 8))
        
        if path:
            # Desenha linha azul para o caminho
            for move in path:
                start_city = self.cities[move[0]]
                end_city = self.cities[move[1]]
                plt.plot([start_city[0], end_city[0]], [start_city[1], end_city[1]], 'b-', alpha=0.7, linewidth=1)
            # Destaca a cidade inicial/final
            plt.plot(x[path[0][0]], y[path[0][0]], 'go', markersize=10, label='Início/Fim')
            plt.legend()

        plt.plot(x, y, 'ro', markersize=6)
        
        # Numeração das cidades
        for i, (xi, yi) in enumerate(self.cities):
            plt.text(xi + 1, yi + 1, str(i), fontsize=9)

        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

# --- BLOCO PRINCIPAL (INTERATIVO) ---

def get_input(text, default, tipo=float):
    valor = input(f"{text} (Padrão: {default}): ")
    if not valor:
        return default
    return tipo(valor)

print("=== CONFIGURAÇÃO DO ACO (ANT SYSTEM) ===")
print("Pressione ENTER para usar o valor padrão sugerido.\n")

# 1. Configuração do Mapa
num_cities = get_input("Número de Cidades (Recomendado > 30 para dificuldade)", 40, int)

# 2. Parâmetros do Algoritmo
n_ants = get_input("Número de Formigas (m)", 20, int)
n_iterations = get_input("Número de Iterações", 100, int)

print("\n--- Parâmetros da Fórmula ---")
alpha = get_input("Alpha (Peso do Feromônio - Histórico)", 1.0, float)
beta = get_input("Beta (Peso da Heurística - Distância)", 2.0, float)
rho = get_input("Rho (Taxa de Evaporação 0.0 a 1.0)", 0.5, float)

# Geração do Problema
print(f"\nGerando mapa aleatório com {num_cities} cidades...")
cities = []
# Sem seed fixa, mapa sempre novo
for i in range(num_cities):
    cities.append((random.randint(0, 100), random.randint(0, 100)))

distances = []
for i in range(num_cities):
    row = []
    for j in range(num_cities):
        if i == j:
            row.append(float('inf'))
        else:
            dist = math.sqrt((cities[i][0]-cities[j][0])**2 + (cities[i][1]-cities[j][1])**2)
            row.append(dist)
    distances.append(row)

# Execução
ant_colony = AntColony(distances, cities, n_ants, n_iterations, rho, alpha, beta)
shortest_path = ant_colony.run()

print("\n" + "="*40)
print(f"RESULTADO FINAL")
print(f"Parâmetros: A={alpha}, B={beta}, Rho={rho}")
print(f"Menor Distância: {shortest_path[1]:.4f}")
print("="*40)