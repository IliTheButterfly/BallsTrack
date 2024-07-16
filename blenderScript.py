import bpy
import socket
from mathutils import Matrix

    
def main():
    # Set up the socket client
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65432        # The port used by the server

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        obj = None
        while True:
            # Receive the data from the external script
            data = s.recv(1024).decode().strip()
            if not data:
                break
            
            # Parse the name and transformation matrix
            name, matrix_str = data.split(":")
            matrix = [float(x) for x in matrix_str.split(",")]
            m2 = []
            for y in range(4):
                row = []
                for x in range(4):
                    i = y*4+x
                    row[x] = matrix[i]
                m2.append(row)

            print(matrix)
            print(m2)
            if obj is None:
                # Create the object
                obj = bpy.data.objects.new(name, None)
                bpy.context.scene.collection.objects.link(obj)
                
            obj.matrix_world = Matrix(m2)

if __name__ == "__main__":
    main()