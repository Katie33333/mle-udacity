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
   def...
   def..
   
 class Shirt(Clothing):
   def...
   
 class Pants(Clothing):
   def...
 ```
