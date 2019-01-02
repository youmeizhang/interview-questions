## Object Oriented Programming Interview Questions
#### 1. OOP
OOPS is abbreviated as Object Oriented Programming system in which programs are considered as a collection of objects. Each object is nothing but an instance of a class.

#### 2. Basic concepts of OOPS
Abstraction, Encapsulation, Inheritance, Polymorphism.

#### 3. Class
A class is simply a representation of a type of object. It is the blueprint/ plan/ template that describes the details of an object.

#### 4. Object
An object is an instance of a class. It has its own state, behaviour, and identity. Objects hold multiple information but classes don’t have any information. Definition of properties and functions can be done in class and can be used by the object. A class has sub-class and an object doesn’t have sub-objects.

#### 5. Encapsulation
It is about binding together the data and functions that manipulate the data and keeps both safe from outside interface and misuse. Encapsulation is an attribute of an object, and it contains all data which is hidden. That hidden data can be restricted to the members of that class. Data encapsulation leads to the important OOP concept of data hiding. Levels are Public, Protected, Private, Internal and Protected Internal.

#### 6. Polymorphism
Polymorphism refers to a programming language’s ability to process objects differently depending on their data type or class. More specifically, it is the ability to redefine methods for derived class. For example, given a base class shape, polymorphism enables the programmer to define different area methods for any number of derived classes, such as circles, rectangles and triangles. No matter what shape an object is, applying the area method to it will return the correct results. Simply, polymorphism takes more than one form.
Object

#### 7. Inheritance
Inheritance is a concept where one class shares the structure and behaviour defined in another class. If inheritance applied on one class is called Single Inheritance, and if it depends on multiple classes, then it is called multiple Inheritance.

#### 8. Abstraction
Data abstraction refers to providing only essential information to the outside world and hiding their background details, i.e., to represent the needed information in program without presenting the details. In order to reduce complexity and increase efficiency.

#### 9. Manipulator
Manipulators are functions specifically designed to be used in conjunction with the insertion (<<) and extraction (>>) operators on stream objects

#### 10. Constructor
A constructor is a method used to initialize the state of an object, and it gets invoked at the time of object creation. It is about creating an object into a known state. Rules for constructor are: Constructor Name should be same as class name. A constructor must have no return type. We don’t require a parameter for constructors.

#### 11. Destructor
A destructor is a method which is automatically called when the object is made out of scope or destroyed. Destructor name is also same as class name but with the tilde symbol before the name.

#### 12. Inline function
An inline function is a technique used by the compilers and instructs to insert complete body of the function wherever that function is used in the program source code. These functions are very short and contain one or two statements.

#### 13. Virtual function
A virtual function is a member function of a class, and its functionality can be overridden in its derived class. This function can be implemented by using a keyword called virtual, and it can be given during function declaration.

#### 14. Friend function
A friend function is a friend of a class that is allowed to access to Public, private or protected data in that same class. If the function is defined outside the class cannot access such information. Friend can be declared anywhere in the class declaration, and it cannot be affected by access control keywords like private, public or protected.

#### 15. Function overloading
Function overloading an as a normal function, but it can perform different tasks. It allows the creation of several methods with the same name which differ from each other by the type of input and output of the function.

#### 16. Operator overloading
In C++, we can make operators to work for user defined classes. For example, we can overload an operator ‘+’ in a class like String so that we can concatenate two strings by just using +.

#### 17. Abstract class
Use Abstract class when there is a 'IS-A' relationship between classes. For example, Lion is a Animal, Cat is an Animal. So, Animal can be an abstract class with common implementation like no. of legs, tail etc. An abstract class is a class which cannot be instantiated. Creation of an object is not possible with an abstract class, but it can be inherited. An abstract class can contain only Abstract method. Java allows only abstract method in abstract class while for other languages allow non-abstract method as well.

#### 18. Ternary Operator
The ternary operator is also known as the conditional operator. This operator consists of three operands and is used to evaluate Boolean expressions. The goal of the operator is to decide, which value should be assigned to the variable. The operator is written as: variable x = (expression) ? value if true: value if false

#### 19. What are different types of arguments?
* Call by Value – Value passed will get modified only inside the function, and it returns the same value whatever it is passed it into the function.
* Call by Reference – Value passed will get modified in both inside and outside the functions and it returns the same or different value.

#### 20. Class and methods
You might define methods inside a class. Outside of a class, methods are called functions. A class is a set of rules you write that govern an object. An object is what a class defines. A method is a bit of code that can be called.

#### 21. Method overriding
Method overriding is a feature that allows a subclass to provide the implementation of a method that overrides in the main class. This will overrides the implementation in the superclass by providing the same method name, same parameter and same return type.

#### 22. Interface
An interface is a collection of an abstract method. If the class implements an inheritance, and then thereby inherits all the abstract methods of an interface.

#### 23. Exception handling
An exception is an event that occurs during the execution of a program. Exceptions can be of any type – Runtime exception, Error exceptions. Those exceptions are adequately handled through exception handling mechanism like try, catch and throw keywords.

#### 24. Overloading and overriding
* Overloading is nothing but the same method with different arguments, and it may or may not return the same value in the same class itself. (same name, different parameters)
* Overriding is the same method names with same arguments and return types associated with the class and its child class. (re-define body of a method of a superclass in a subclass to change behaviour of a method)

#### 25. Access modifiers
Access modifiers determine the scope of the method or variables that can be accessed from other various objects or classes. There are 5 types of access modifiers, and they are as follows: Private, Protected, Public, Friend, Protected Friend.

#### 26. You can call the base method without creating an instance

#### 27. Difference between new and override
The new modifier instructs the compiler to use the new implementation instead of the base class function. Whereas, Override modifier helps to override the base class function.

#### 28. Types of constructors
* Default Constructor – With no parameters.
* Parametric Constructor – With Parameters. Create a new instance of a class and also passing arguments simultaneously.
* Copy Constructor – Which creates a new object as a copy of an existing object.

#### 29. ‘this’ pointer
THIS pointer refers to the current object of a class. THIS keyword is used as a pointer which differentiates between the current object with the global object. Basically, it refers to the current object.

#### 30. Difference between structure and a class
Structure default access type is public , but class access type is private. A structure is used for grouping data whereas class can be used for grouping data and methods. Structures are exclusively used for data, and it doesn’t require strict validation , but classes are used to encapsulates and inherit data which requires strict validation.

#### 31. Default access modifier in a class is private
Pure virtual function: A pure virtual function is a function which can be overridden in the derived class but cannot be defined. A virtual function can be declared as Pure by using the operator = 0.

#### 32. Run time polymorphism
Dynamic or Run time polymorphism is also known as method overriding in which call to an overridden function is resolved during run time, not at the compile time. It means having two or more methods with the same name, same signature but with different implementation.

#### 33. Copy constructor
This is a special constructor for creating a new object as a copy of an existing object. There will always be only one copy constructor that can be either defined by the user or the system.

#### 34. What does the keyword virtual represented in the method definition
It means, we can override the method.

#### 35. Base class, super class and sub-class
* The base class is the most generalized class, and it is said to be a root class. 
* A Sub class is a class that inherits from one or more base classes.
* The superclass is the parent class from which another class inherits.

#### 36. Which keyword can be used for overloading?
Operator keyword is used for overloading.

#### 37. Which OOPS concept is used as reuse mechanism?
Inheritance is the OOPS concept that can be used as reuse mechanism.

#### 38. Which OOPS concept exposes only necessary information to the calling functions?
Encapsulation

#### 39. Which operators cannot be overloaded?
Scope Resolution Operator

#### 40. Static method can not use non static members

#### 41. The advantages of object oriented programming
5 features of OOP: inheritance, polymorphism, encapsulation, data abstraction
* code reuse and recycling
* encapsulation: once object is created, knowledge of its implementation is not necessary for its use. In older programs, coders needed to understand the details of a piece of code before using it. Objects have the ability to hide certain parts of themselves from programmers
* design benefit: it forces software engineer to go through an extensive planning phrase.
* software maintenance: easier to modify and maintain
