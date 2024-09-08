import javalang
def get_java_tree(java_code):
    java_code="public class HelloWorld {"+java_code+"}"


    def traverse_ast(node, visited, result):
        if node not in visited:
            visited.add(node)
            result.append(node.__class__.__name__)
            for _, child_node in node.filter(javalang.tree.Node):
                traverse_ast(child_node, visited, result)


    def get_java_ast(code):


        tree = javalang.parse.parse(code)

        result = []
        visited = set()

        traverse_ast(tree, visited, result)


        # for node_class in result:
        #     print(node_class)
        # print(result)
        return " ".join(result)

    return get_java_ast(java_code)
if __name__ == "__main__":
    java_code="""
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
    """
    print(get_java_tree(java_code))
    java_code="""protected void modify ( Transaction t ) { try { this . lock . writeLock ( ) . lock ( ) ; t . perform ( ) ; } finally { this . lock . writeLock ( ) . unlock ( ) ; } }"""
    print(get_java_tree(java_code))
    python_code = """
    def greet(name):
        print("Hello, " + name)

    for i in range(3):
        greet("User" + str(i))
        """
    print(get_java_tree(java_code))
    import time

    start_time = time.time()


    print(get_java_tree(java_code))

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

