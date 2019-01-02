
## CS Basic Interview Questions

### Python
#### 1. Difference between list and tuples
* list: mutable, they can be edited, slower than tuples
* tuple: immutable, can not be edited, faster than lists

#### 2. key features
* interpreted language: does not need to be compiled before it is run
* dynamically typed: don’t need to state the variable type before when you declare them
* functions are first-class objects
* writing python code is quick but running is slower than compiled language, but it allows the inclusion of C based extensions so bottlenecks can be optimized away such as numpy package
* python is used in many spheres such as web automation scientific modelling big data

#### 3. First class functions
* functions are objects: assign the function to a variable
* functions can be passed as arguments to other functions: greet(shout), greet(whisper) 
* functions can return another function

#### 4. Shallow copy and deep copy
* shallow copy: copy the reference pointers just like it copies the values, changes can affect the original copy. If we change any nested objects in original data, then copy will also be changed. If not nested objects, then copy is not affected, like append.
* deep copy: store the values that are already copied, it does not copy the reference pointers to the objects. It makes the reference to an object and the new object that is pointed by some other object gets stored. If makes changes to original data, then the deep copy one is not affected at all

#### 5. How is Multithreading achieved in Python
* Python has a multi-threading package but if you want to multi-thread to speed your code up, then it’s usually not a good idea to use it.
* Python has a construct called the Global Interpreter Lock (GIL). The GIL makes sure that only one of your ‘threads’ can execute at any one time. A thread acquires the GIL, does a little work, then passes the GIL onto the next thread. 
* This happens very quickly so to the human eye it may seem like your threads are executing in parallel, but they are really just taking turns using the same CPU core.
* All this GIL passing adds overhead to execution. This means that if you want to make your code run faster then using the threading package often isn’t a good idea.

#### 6. Ternary operations
The Ternary operator is the operator that is used to show the conditional statements. This consists of the true or false values with a statement that has to be evaluated for it. Such as: min = a if a < b else b 

#### 7. How is memory managed in Python?
* Memory management in python is managed by Python private heap space. All Python objects and data structures are located in a private heap. The programmer does not have access to this private heap. The python interpreter takes care of this instead. 
* The allocation of heap space for Python objects is done by Python’s memory manager. The core API gives access to some tools for the programmer to code. 
* Python also has an inbuilt garbage collector, which recycles all the unused memory and so that it can be made available to the heap space.

#### 8. Inheritance
one child class to gain attributes and methods from super class. 
* single inheritance: acquires the members of a single super class
* multi-level inheritance: derived class 2 is inherited from derived class 1 and d1 is inherited from base class 1

#### 9. Flask
Microframework based on werkzeug, jinja2 as dependencies, so it has little dependencies on external libraries. A session basically allows you to remember information from one request to another. In a flask, a session uses a signed cookie so the user can look at the session contents and modify. The user can modify the session if only it has the secret key Flask.secret_key.

#### 10. dir()
The dir() function is used to display the defined symbols

#### 11. Whenever Python exits, why isn’t all the memory de-allocated
* Whenever Python exits, especially those Python modules which are having circular references to other objects or the objects that are referenced from the global namespaces are not always de-allocated or freed. 
* It is impossible to de-allocate those portions of memory that are reserved by the C library.
* On exit, because of having its own efficient clean up mechanism, Python would try to de-allocate/destroy every other object.

#### 12. What does this mean: *args, **kwargs? And why would we use it
We use *args when we aren’t sure how many arguments are going to be passed to a function, or if we want to pass a stored list or tuple of arguments to a function. **kwargsis used when we don’t know how many keyword arguments will be passed to a function, or it can be used to pass the values of a dictionary as keyword arguments. The identifiers args and kwargs are a convention, you could also use *bob and **billy but that would not be wise.

#### 13. Write a one-liner that will count the number of capital letters in a file. Your code should work even if the file is too big to fit in memory.
count = sum(1 for line in fh for character in line if character.isupper())

#### 14. Randomize items in a list in place
```Python
from random import shuffle
s = [1,2,3,4]
shuffle(s)
```

#### 15. Sorting function for numerical dataset
```Python
list = [“1”, “2”, “4”]
list = [int(i) for I in list]
list = sorted(list)
```

#### 16. sub() and subn()
* sub() – finds all substrings where the regex pattern matches and then replace them with a different string
* subn() – it is similar to sub() and also returns the new string along with the no. of replacements.

#### 17. Random
* randrange(a, b): [a, b)
* uniform(a, b): floating point number that is defined in the range of [a, b)
* normalvariate(mean, sdev): used for the normal distribution mu is mean and side is sigma the is used for standard deviation
Range and xrange
* range: return python list object
* xrange: return an xrange object, used when the range is really large such as one billion to avoid memory error

#### 18. Pickling and unpickling
Pickle module accepts any Python object and converts it into a string representation and dumps it into a file by using dump function, this process is called pickling. While the process of retrieving original Python objects from the stored string representation is called unpickling.

#### 19. Save an image with url known
```Python
import urllib.request
urllib.request.urlretrieve(“URL”, “filename.jpg”)
```  

#### 20. Map function
execute the function (first argument) on all the elements of the iterable (second argument)

#### 21. get indices of N maximum values in a NumPy array?
```Python
import numpy as np
arr = np.array([1, 3, 2, 4, 5])
print(arr.argsort()[-3:][::-1])
```

#### 22. Calculate percentiles 
p = np.percentile(a, 50) #50%

#### 23. List
general-purpose containers, efficient insertion, deletion, appending, concatenation.  Limitation: don’t support vectorized operations, such as element wise addition and multiplication. List can contain different types of elements, so need to execute type dispatching code when operating on each element.

#### 24. Numpy
convenient, lots of vector and matrix operations for free. It is faster and get lots of built in functions for fast searching, linear algebra and histogram

#### 25. Explain the use of decorators
Decorators in Python are used to modify or inject code in functions or classes. Using decorators, you can wrap a class or function method call so that a piece of code can be executed before or after the execution of the original code. Decorators can be used to check for permissions, modify or track the arguments passed to a method, logging the calls to a specific method, etc

#### 26. 3D plot
Like 2D plotting, 3D graphics is beyond the scope of NumPy and SciPy, but 3D plot is provided by matplotlib

#### 27. Dictionary is created by specifying keys and values, so d = {} is not creating a dictionary

#### 28. When will the else part of try-except-else be executed?
when no exception occurs

#### 29. What are the tools that help to find bugs or perform static analysis?
PyChecker is a static analysis tool that detects the bugs in Python source code and warns about the style and complexity of the bug. Pylint is another tool that verifies whether the module meets the coding standard

#### 30. How are arguments passed by value or by reference?
Everything in Python is an object and all variables hold references to the objects. The references values are according to the functions; as a result you cannot change the value of the references. However, you can change the objects if it is mutable

#### 31. What is Dict and List comprehensions are?
They are syntax constructions to ease the creation of a Dictionary or List based on existing iterable, such as: [x for x in old_list] [ x for x in range(0, 15)]

#### 32. Built-in types in python
* mutable: list, sets, dictionaries
* immutable: strings, tuples, numbers

#### 33. Namespace
like a dic, key is the name of the variables, value is the real value of that variable. Python中，每个函数都有一个自己的命名空间，叫做local namespace，它记录了函数的变量。python中，每个module有一个自己的命名空间，叫做global namespace，它记录了module的变量，包括 functions, classes 和其它imported modules，还有 module级别的 变量和常量。还有一个build-in 命名空间，可以被任意模块访问，这个build-in命名空间中包含了build-in function 和 exceptions。当python中的某段代码要访问一个变量x时，python会在所有的命名空间中寻找这个变量，查找的顺序为: local, global, built-in (error: namerror)

#### 34. Lambda
It is a single expression anonymous function often used as inline function.

#### 35. Why lambda forms in python does not have statements?
A lambda form in python does not have statements as it is used to make new function object and then return them at runtime

#### 36. What is unittest in Python?
A unit testing framework in Python is known as unittest. It supports sharing of setups, automation testing, shutdown code for tests, aggregation of tests into collections etc.

#### 37. Slicing
A mechanism to select a range of items from sequence types like list, tuple, strings etc. is known as slicing.

#### 38. Generator
L = [x for x in range(10)], g = (x for x in range(10))
* yield: generator, when it calls .next() return results when meets yield, next time will continue from yield
The way of implementing iterators are known as generators. It is a normal function except that it yields expression in the function.

#### 39. Docstring
A Python documentation string is known as docstring, it is a way of documenting Python functions, modules and classes

#### 40. What is module and package in Python?
In Python, module is the way to structure program. Each Python program file is a module, which imports other modules like objects and attributes. The folder of Python program is a package of modules. A package can have modules or subfolders.

#### 41. How can you share global variables across modules?
To share global variables across modules within a single program, create a special module. Import the config module in all modules of your application. The module will be available as a global variable across modules.

#### 42. Explain how can you make a Python Script executable on Unix?
Script file's mode must be executable and the first line must begin with # ( #!/usr/local/bin/python)

#### 43. Delete a file in Python?
By using a command os.remove (filename) or os.unlink(filename)

#### 44. Mention five benefits of using Python?
* Python comprises of a huge standard library for most Internet platforms like Email, HTML, etc.
* Python does not require explicit memory management as the interpreter itself allocates the memory to new variables and free them automatically
*	Provide easy readability due to use of square brackets
*	Easy-to-learn for beginners
*	Having the built-in data types saves programming time and effort from declaring variables

#### 45. Explain what is the common way for the Flask script to work?
Either it should be the import path for your application Or the path to a Python file

#### 46. Is Flask an MVC model
Basically, Flask is a minimalistic framework which behaves same as MVC framework. So MVC is a perfect fit for Flask

#### 47. Explain database connection in Python Flask?
Flask supports database powered application (RDBS). Such system requires creating a schema, which requires piping the shema.sql file into a sqlite3 command. So you need to install sqlite3 command in order to create or initiate the database in Flask.

#### 48. Flask allows to request database in three ways
* before_request() : They are called before a request and pass no arguments
*	after_request() : They are called after a request and pass the response that will be sent to the client
*	teardown_request(): They are called in situation when exception is raised, and response are not guaranteed. They are called after the response been constructed. They are not allowed to modify the request, and their values are ignored.

#### 49. Name the File-related modules in Python?
Python provides libraries / modules with functions that enable you to manipulate text files and binary files on file system. Using them you can create files, update their contents, copy, and delete files. The libraries are : os, os.path, and shutil. Here, os and os.path – modules include functions for accessing the filesystem shutil – module enables you to copy and delete the files.

#### 50. Explain the use of with statement?
In python generally “with” statement is used to open a file, process the data present in the file, and also to close the file without calling a close() method. “with” statement makes the exception handling simpler by providing cleanup activities.
* General form of with: with open(“filename”, “mode”) as file-var: processing statements...
* note: no need to close the file by calling close() upon file-var.close()

#### 51. file processing modes
r, w, rw, text file, add option ’t’, so: rt, wt, rwt, binary file, add option ‘b’

#### 52. Sequence
str, list, tuple, unicode, byte array, xrange, and buffer

#### 53. Display the contents of text file in reverse order?
```Python
For line in reversed(list(open(“file-name”,”r”))): 
  print(line)
```

#### 54. Name few Python modules for Statistical, Numerical and scientific computations?
NumPy – this module provides an array/matrix type, and it is useful for doing computations on arrays. scipy – this module provides methods for doing numeric integrals, solving differential equations, etc pylab – is a module for generating and saving plots
	
#### 55. What is multithreading?
It means running several different programs at the same time concurrently by invoking multiple threads. Multiple threads within a process refer the data space with main thread and they can communicate with each other to share information more easily.Threads are light-weight processes and have less memory overhead. Threads can be used just for quick task like calculating results and also running other processes in the background while the main program is running.

#### 56. Methods that are used to implement Functionally Oriented Programming in Python?
* filter() – enables you to extract a subset of values based on conditional logic.
*	map() – it is a built-in function that applies the function to each item in an iterable.
*	reduce() – repeatedly performs a pair-wise reduction on a sequence until a single value is computed. reduce把结果继续和序列的下一个元素做累积计算

#### 57. The argument given for the set must be an iterable. Not correct: set([[1,2],[3,4]])
