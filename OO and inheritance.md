Notes
**magic methods** override and customize default python behavior (__init__ method, __repr__ representation method, for example)
  * __add__ magic method overrides the behavior of the plus sign.  This function has two inputs:
  ```
  def __add__(self, other):
  ...
  ```
  * self refers to the object to the left of the plus sign.  Other refers to the object on the right of the plus sign
  
  **inheritance**
  
  In the shirt class example, we could code a parent clothing class with attributes color, size, style, price and methods change_price and discount_price and let multiple classes (pants, shirts, socks, dresses) could inherit the attributes and methodes of the clothing class.
  
 ```
 class Clothing:
   def __init__(self, color, style, size, style, price):
     self.color = color
     self.style = style
     self.size = size
     self.style = style
     self.price = price
     
   def change_price(self, price):
     self.price = price 
     
   def calculate_discount(self, discount):
     return self.price * (1 - discount)
   
 class Shirt(Clothing):
   def __init__(self, color, style, size, style, price, long_or_short):
     
     Clothing.__init__(self, color, size, style, price)
     self.long_or_short = long_or_short
     
   def double_price(self):
     self.price = 2*self.price
     
 class Pants(Clothing):
   def...
 ```

* Now the Shirt and Pants classes inherit attributes and methodes of the Clothing class.
* The shirt object initializes itself using the Clothing object's init method


**How to make a Python Package**
* The __init__.py file tells Python this folder contains a package.  The code inside an init file gets run whenever you import a package inside of a Python program.  It can import a class from a module for example.
* A setup.py file is at the same level as a distributions folder and is necessary for pip installing.  It'll contain metadata about the package like the pkg name, version, description etc
* to install, go to teh dir containing the setup.py file and type: pip install .  (The dot tells pip to look for the setup file in the current folder.)
* How to find out where the package was installed:  If you did an import of distributions, then you type `distributions.__file__`

