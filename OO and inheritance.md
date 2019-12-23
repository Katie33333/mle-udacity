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
