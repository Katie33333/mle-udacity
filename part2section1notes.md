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

