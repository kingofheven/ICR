import javalang

def get_java_types(java_code):
    if type(java_code)==list:
        java_code=" ".join(java_code)
    if type(java_code)==str:
        java_code=" ".join(java_code.split())

    tokens = list(javalang.tokenizer.tokenize(java_code))

    # for token in tokens:
    #     print(f"Type: {token.__class__.__name__}, Value: {token.value}")
    return [token.__class__.__name__ for token in tokens]

if __name__ == "__main__":

    java_code = """
    public class HelloWorld {
        public static void main(String[] args) {
            String greeting = "Hello, World!";
            System.out.println(greeting);
        }
    }
    """

    get_java_types(java_code)
