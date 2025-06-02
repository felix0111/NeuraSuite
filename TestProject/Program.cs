namespace TestProject {

    public class Program {

        static void Main(string[] args) {

            while (true) {
                Console.Clear();
                Console.WriteLine("Enter 'neat' to test the original NEAT implementation. Enter 'neatex' to test the expanded NEAT implementation. \n");

                var input = Console.ReadLine();
                switch (input) {
                    case "neat":
                        NeatTest.RunTest();
                        break;
                    case "neatex":
                        NeatExpandedTest.RunTest();
                        break;
                    default:
                        return;
                }
            }
            
        }
    }
}