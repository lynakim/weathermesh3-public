RESEARCH DIRECTIONS
- normalization: doing zero mean unit variance is kinda fucky, especially with specific humidity, me no likey. RH might be a stopgap, there also might be better ways to go about it
- encoding winds: dot products to align it with the mesh-to-mesh vectors?

TODO:
- interpolation doesn't handle the 359.75...360 range properly (it's merely extrapolated)

A0: ERA-5 at t0
A1: ERA-5 at t1
e: Encoder (Linear + Conv)
d: Decoder (Conv + Linear)
r: Reality
f: Processor (Transformer)
B0: Hidden state for t0 
B1: Hidden state for t1


 A0   -r->  A1 
 |          ^  
 |          |     
 e          d 
 |          |  
 v          |   
 B0   -f->  B1 
  
  
  
   \       /
    '-dd -'
       |
       | 
       v 
    (A1-A0)

dd: Delta Decoder, takes in B0 and B1 and outputs 
 


 
```python
# 3hr
xA = A(x)  # encode
y_3 = x + C_3(xA,B(xA))

# 6hr
xA = A(x)  # encode
y_6 = C_3(xA,B^2(xA))

# 9hr
xA = A(x)  # encode
xB2 = B^2(xA) # +6hrs
y = x + C_6(xA, B^2(xA)) + C_3(xB2, B(xB2)) # cursed decoding, sum up deltas
```

