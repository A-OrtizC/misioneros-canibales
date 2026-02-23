
# missionaries_cannibals_solver.py
# Solución al acertijo de los Misioneros y Caníbales usando BFS
# Representación de estados: (M_left, C_left, boat)
# boat = 0 -> izquierda, 1 -> derecha

from collections import deque

START = (3, 3, 0)
GOAL = (0, 0, 1)

# Movimientos posibles: (misioneros, caníbales)
MOVES = [(1,0), (2,0), (0,1), (0,2), (1,1)]

def is_valid(state):
    Ml, Cl, boat = state
    Mr = 3 - Ml
    Cr = 3 - Cl

    # Rango válido
    if not (0 <= Ml <= 3 and 0 <= Cl <= 3):
        return False

    # Restricciones de seguridad
    if Ml > 0 and Cl > Ml:
        return False
    if Mr > 0 and Cr > Mr:
        return False

    return True

def successors(state):
    Ml, Cl, boat = state
    succs = []

    # Dirección del movimiento
    if boat == 0:   # izquierda -> derecha
        sign = -1
    else:           # derecha -> izquierda
        sign = 1

    for dm, dc in MOVES:
        new_state = (Ml + sign*dm, Cl + sign*dc, 1 - boat)
        if is_valid(new_state):
            succs.append((new_state, (dm, dc)))

    return succs

def bfs(start, goal):
    queue = deque([start])
    parent = {start: None}
    move_used = {start: None}

    while queue:
        state = queue.popleft()
        if state == goal:
            return parent, move_used

        for new_state, move in successors(state):
            if new_state not in parent:
                parent[new_state] = state
                move_used[new_state] = move
                queue.append(new_state)

    return None, None

def reconstruct_path(parent, move_used, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append((cur, move_used[cur]))
        cur = parent[cur]
    path.reverse()
    return path

def main():
    parent, move_used = bfs(START, GOAL)
    if parent is None:
        print("No se encontró solución.")
        return

    path = reconstruct_path(parent, move_used, GOAL)

    print("\nSolución encontrada:\n")
    for i, (state, move) in enumerate(path):
        Ml, Cl, boat = state
        side = "izquierda" if boat == 0 else "derecha"
        if move is None:
            print(f"Paso {i}: Estado {state}, barco en {side} (inicio)")
        else:
            dm, dc = move
            print(f"Paso {i}: Estado {state}, barco en {side}, movimiento previo = {dm}M, {dc}C")

    print("\nSecuencia resumida de cruces:")
    for (state, move) in path[1:]:
        dm, dc = move
        print(f"{dm} misioneros, {dc} caníbales")

if __name__ == "__main__":
    main()
