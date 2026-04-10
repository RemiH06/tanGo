from neo4j import GraphDatabase
import networkx as nx

class TrafficGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="12345678"):
        self.graph = nx.DiGraph()
        # Inicializamos el driver de Neo4j
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def add_intersection(self, intersection):
        self.graph.add_node(intersection.semaphore_id, data=intersection)

    def add_road(self, road):
        self.graph.add_edge(road.from_intersection, road.to_intersection, data=road)

    def sync_to_neo4j(self):
        """Sincroniza el grafo de NetworkX a la base de datos Neo4j."""
        with self.driver.session() as session:
            # 1. Crear/Actualizar Nodos (Intersecciones)
            for node_id, attrs in self.graph.nodes(data=True):
                inter = attrs['data']
                session.run("""
                    MERGE (i:Intersection {semaphore_id: $id})
                    SET i.location = point({latitude: $lat, longitude: $lng}),
                        i.current_phase = $phase,
                        i.cycle_duration = $duration
                """, id=inter.semaphore_id, lat=inter.location[0], 
                     lng=inter.location[1], phase=inter.current_phase, 
                     duration=inter.cycle_duration)

            # 2. Crear/Actualizar Relaciones (RoadSegments)
            for u, v, attrs in self.graph.edges(data=True):
                road = attrs['data']
                session.run("""
                    MATCH (a:Intersection {semaphore_id: $from_id})
                    MATCH (b:Intersection {semaphore_id: $to_id})
                    MERGE (a)-[r:ROADS_TO]->(b)
                    SET r.current_speed = $speed,
                        r.vehicle_density = $density
                """, from_id=u, to_id=v, speed=road.current_speed, 
                     density=road.vehicle_density)
        print("Sincronización con Neo4j completada.")

# --- PRUEBA DE CONEXIÓN ---
if __name__ == "__main__":
    # Ajusta tu password de Neo4j aquí
    tango_db = TrafficGraph(password="admin1234") 
    
    # ... (aquí iría la creación de nodos de la prueba anterior) ...
    
    try:
        # tango_db.sync_to_neo4j()
        print("Prueba de sync lista")
    finally:
        tango_db.close()


import networkx as nx

class Intersection:
    def __init__(self, semaphore_id, lat, lng, base_weight=20):
        self.semaphore_id = semaphore_id
        self.location = (lat, lng)
        self.current_phase = "rojo"
        self.cycle_duration = 60
        self.base_weight = base_weight # Ejemplo: 20 para calle residencial

    def adjust_cycle(self, traffic_flow, weather, hour):
        # Aquí iría tu lógica con el motor de pesos
        pass

    def __repr__(self):
        return f"Intersection({self.semaphore_id}, phase={self.current_phase})"


class RoadSegment:
    def __init__(self, from_id, to_id, base_weight=50):
        self.from_intersection = from_id
        self.to_intersection = to_id
        self.current_speed = 40
        self.vehicle_density = 0
        self.base_weight = base_weight # Ejemplo: 50 para avenida secundaria

    def congestion_level(self):
        # Lógica para determinar congestión
        return self.vehicle_density / self.base_weight


class TrafficGraph:
    def __init__(self):
        # Inicializamos el grafo dirigido (las calles tienen sentido)
        self.graph = nx.DiGraph()

    def add_intersection(self, intersection: Intersection):
        # Agregamos el nodo usando el ID como clave y el objeto como atributo
        self.graph.add_node(intersection.semaphore_id, data=intersection)

    def add_road(self, road: RoadSegment):
        # Agregamos la arista con sus datos
        self.graph.add_edge(road.from_intersection, road.to_intersection, data=road)

    def get_nodes(self):
        return [data['data'] for node, data in self.graph.nodes(data=True)]

    def get_edges(self):
        return [data['data'] for u, v, data in self.graph.edges(data=True)]

    def get_info(self):
        return f"Nodos (Intersecciones): {self.graph.number_of_nodes()} | Aristas (Calles): {self.graph.number_of_edges()}"


# --- ZONA DE PRUEBAS (Generador chiquito) ---
def correr_prueba_generador():
    print("Iniciando prueba del generador de tanGo...")
    
    # 1. Crear la instancia del grafo
    city_graph = TrafficGraph()

    # 2. Crear un par de intersecciones (Nodos)
    int_a = Intersection(semaphore_id="SEM-001", lat=20.659, lng=-103.349)
    int_b = Intersection(semaphore_id="SEM-002", lat=20.660, lng=-103.350)
    int_c = Intersection(semaphore_id="SEM-003", lat=20.661, lng=-103.351)

    city_graph.add_intersection(int_a)
    city_graph.add_intersection(int_b)
    city_graph.add_intersection(int_c)

    # 3. Conectar las intersecciones con calles (Aristas)
    # Calle de A -> B
    road_ab = RoadSegment(from_id="SEM-001", to_id="SEM-002", base_weight=80) 
    # Calle de B -> C
    road_bc = RoadSegment(from_id="SEM-002", to_id="SEM-003", base_weight=80)

    city_graph.add_road(road_ab)
    city_graph.add_road(road_bc)

    # 4. Validar que todo exista
    print("\nResumen del grafo:")
    print(city_graph.get_info())

    print("\nDetalle de Intersecciones:")
    for node in city_graph.get_nodes():
        print(f" - {node}")

if __name__ == "__main__":
    correr_prueba_generador()