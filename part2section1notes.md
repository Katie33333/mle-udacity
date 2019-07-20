* General Software Engineering Practics
* Object-oriented programming
* web development


Welcome To Software Engineering Practices Part I
In this lesson, you'll learn about the following practices of software engineering and how they apply in data science.

* Writing clean and modular code
* Writing efficient code
* Code refactoring
* Adding meaningful documentation
* Using version control
* In the lesson following this one (Part II) you'll also learn these software engineering practices:

Testing
Logging
Code reviews

**Steps**
* Get code functioning
* Refactor - clean code
* Refactor - efficient code (reduce run time and space in memory)


**Version Control**
* git log - show the history of your commits
* find the commit you're looking for
* git checkout ccac636b8923485798429cg...


**Testing, Logging, Code Reviews**
* "Assert" will compare the actual and expected result of a unit test
* You don't want unit tests to stop the program.  You want to run all unit tests to know which failed and which were successful.
* pytest - (pip install -U pytest), create a file where you'll write a function for your unit test.
  * type "pytest" while in that directory.  It will run all tests and return a dot for passed and F for fail
* One assert statement per test
* test files and test functions need to be named as test_....
* test driven development - write tests first with every possible scenario you can think of, and then the implementation of the code

**Logging**
DEBUG - level you would use for anything that happens in the program.
ERROR - level to record any error that occurs
INFO - level to record all actions that are user-driven or system specific, such as regularly scheduled operations

**Code Review**
Questions to Ask Yourself When Conducting a Code Review
First, let's look over some of the questions we may ask ourselves while reviewing code. These are simply from the concepts we've covered in these last two lessons!

* **Is the code clean and modular?**
* Can I understand the code easily?
* Does it use meaningful names and whitespace?
* Is there duplicated code?
* Can you provide another layer of abstraction?
* Is each function and module necessary?
* Is each function or module too long?
* **Is the code efficient?**
* Are there loops or other steps we can vectorize?
* Can we use better data structures to optimize any steps?
* Can we shorten the number of calculations needed for any steps?
* Can we use generators or multiprocessing to optimize any steps?
* **Is documentation effective?**
* Are in-line comments concise and meaningful?
* Is there complex code that's missing documentation?
* Do function use effective docstrings?
* Is the necessary project documentation provided?
* **Is the code well tested?**
* Does the code high test coverage?
* Do tests check for interesting cases?
* Are the tests readable?
* Can the tests be made more efficient?
* **Is the logging effective?**
* Are log messages clear, concise, and professional?
* Do they include all relevant and useful information?
* Do they use the appropriate logging level?
