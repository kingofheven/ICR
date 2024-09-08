from get_python_tree import get_python_tree
from get_java_tree import get_java_tree
from get_nl_tree import get_nl_tree
def get_trees(text):
    trees=""
    try:
        trees = get_java_tree(text)
    except:
        try:
            trees=get_python_tree(text)
        except:
            trees=get_nl_tree(text)
    return trees
if __name__ == "__main__":
    import time

    start_time = time.time()


    print(get_trees("""
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
        """))

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
