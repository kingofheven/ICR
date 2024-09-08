import spacy
nlp = spacy.load("en_core_web_sm")
def get_nl_tree(text):

    doc = nlp(text)

    def preorder_traversal(node, level=0):
        result = f"{node.pos_} "
        for child in node.children:
            result += preorder_traversal(child, level + 1)
        return result

    def generate_syntax_trees(doc):
        result = ""
        for sentence in doc.sents:
            root = sentence.root
            result += preorder_traversal(root)
        return result

    syntax_trees = generate_syntax_trees(doc)
    return syntax_trees
if __name__ == "__main__":
    text = '''what a nice day!'''

    print(get_nl_tree(text))
    import time

    start_time = time.time()


    print(get_nl_tree(text))

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
