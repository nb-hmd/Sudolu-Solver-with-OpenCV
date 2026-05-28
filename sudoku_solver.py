def solve_sudoku(board):
    if not is_board_valid(board):
        return False

    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]

    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == 0:
                continue
            box_i = (r // 3) * 3 + (c // 3)
            rows[r].add(val)
            cols[c].add(val)
            boxes[box_i].add(val)

    return _solve_with_sets(board, rows, cols, boxes)


def is_board_valid(board):
    if len(board) != 9:
        return False
    for row in board:
        if len(row) != 9:
            return False
        for val in row:
            if not isinstance(val, int):
                return False
            if val < 0 or val > 9:
                return False

    for i in range(9):
        seen_row = set()
        seen_col = set()
        for j in range(9):
            rv = board[i][j]
            cv = board[j][i]
            if rv != 0:
                if rv in seen_row:
                    return False
                seen_row.add(rv)
            if cv != 0:
                if cv in seen_col:
                    return False
                seen_col.add(cv)

    for box_r in range(0, 9, 3):
        for box_c in range(0, 9, 3):
            seen = set()
            for r in range(box_r, box_r + 3):
                for c in range(box_c, box_c + 3):
                    v = board[r][c]
                    if v == 0:
                        continue
                    if v in seen:
                        return False
                    seen.add(v)

    return True


def _find_best_cell(board, rows, cols, boxes):
    best_r = None
    best_c = None
    best_box = None
    best_candidates = None

    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                continue
            box_i = (r // 3) * 3 + (c // 3)
            candidates = [
                n
                for n in range(1, 10)
                if n not in rows[r] and n not in cols[c] and n not in boxes[box_i]
            ]
            if not candidates:
                return (r, c, box_i, [])
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_r, best_c, best_box, best_candidates = r, c, box_i, candidates
                if len(best_candidates) == 1:
                    return (best_r, best_c, best_box, best_candidates)

    if best_candidates is None:
        return None
    return (best_r, best_c, best_box, best_candidates)


def _solve_with_sets(board, rows, cols, boxes):
    cell = _find_best_cell(board, rows, cols, boxes)
    if cell is None:
        return True

    r, c, box_i, candidates = cell
    if not candidates:
        return False

    for num in candidates:
        board[r][c] = num
        rows[r].add(num)
        cols[c].add(num)
        boxes[box_i].add(num)

        if _solve_with_sets(board, rows, cols, boxes):
            return True

        boxes[box_i].remove(num)
        cols[c].remove(num)
        rows[r].remove(num)
        board[r][c] = 0

    return False


def valid(board, num, pos):
    """
    Returns if the attempted move is valid
    :param board: current board
    :param num: number to check
    :param pos: (row, col)
    :return: bool
    """
    # Check row
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x*3, box_x*3 + 3):
            if board[i][j] == num and (i,j) != pos:
                return False

    return True


def find_empty(board):
    """
    finds an empty space in the board
    :param board: partially complete board
    :return: (int, int) row col
    """
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)  # row, col

    return None

def print_board(board):
    for i in range(len(board)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")

        for j in range(len(board[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")
