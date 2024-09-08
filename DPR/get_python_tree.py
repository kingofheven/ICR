import ast

def get_python_tree(code):
    node = ast.parse(code)
    result = type(node).__name__

    for child_node in ast.iter_child_nodes(node):
        result += ' ' + get_python_tree(child_node)

    return result



if __name__ == "__main__":
    code = """1+1\n1+1"""
    print(get_python_tree(code))
    import time

    start_time = time.time()


    print(get_python_tree(code))

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
