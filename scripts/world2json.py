import xml.etree.ElementTree as ET
import json

def parse_world_file(world_file):
    """
    Procesa un archivo .world y extrae obstáculos con su posición y geometría,
    ignorando elementos irrelevantes como el suelo.
    """
    tree = ET.parse(world_file)
    root = tree.getroot()

    obstacles = []

    # Buscar modelos en el archivo
    for model in root.findall(".//model"):
        name = model.get("name")

        # Ignorar el suelo o elementos irrelevantes por su nombre
        if name in ["ground_plane", "room", "sun"]:
            continue  # Saltar estos modelos

        # Verificar cada <collision> dentro del modelo
        for collision in model.findall(".//collision"):
            geometry = collision.find(".//geometry")
            pose_tag = collision.find("pose")
            
            # Si no hay pose, usar posición por defecto
            if pose_tag is not None:
                pose = list(map(float, pose_tag.text.split()))
            else:
                pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            x, y, z = pose[:3]

            # Determinar el tipo de geometría
            size = None
            if geometry is not None:
                if geometry.find("box") is not None:
                    size = geometry.find("box/size").text.split()
                    size = list(map(float, size))
                elif geometry.find("cylinder") is not None:
                    radius = float(geometry.find("cylinder/radius").text)
                    length = float(geometry.find("cylinder/length").text)
                    size = [2 * radius, 2 * radius, length]  # Aproximar como un cubo
                elif geometry.find("sphere") is not None:
                    radius = float(geometry.find("sphere/radius").text)
                    size = [2 * radius, 2 * radius, 2 * radius]  # Aproximar como un cubo

            # Agregar obstáculo si tiene tamaño válido
            if size:
                obstacles.append({
                    "name": name,
                    "pose": [x, y, z],
                    "size": size
                })

    return obstacles


def generate_rrt_input(world_file, output_file, bounds):
    """
    Genera un archivo JSON para el RRT* a partir de un archivo .world.
    """
    # Procesar el archivo .world
    obstacles = parse_world_file(world_file)

    # Crear la estructura del JSON
    rrt_data = {
        "bounds": bounds,  # Límites del espacio de planificación
        "obstacles": obstacles  # Obstáculos extraídos
    }

    # Guardar como archivo JSON
    with open(output_file, "w") as f:
        json.dump(rrt_data, f, indent=4)
    print(f"Archivo JSON generado: {output_file}")

# Ruta al archivo .world
world_file = "/home/alvmorcal/robmov_ws/src/quadrotor_planner/worlds/habitacion_pruebas.world"

# Límites del espacio (xmin, xmax, ymin, ymax, zmin, zmax)
bounds = {
    "x": [-10, 10],
    "y": [-10, 10],
    "z": [0, 20]
}

# Ruta del archivo de salida
output_file = "world.json"

# Generar archivo JSON
generate_rrt_input(world_file, output_file, bounds)

