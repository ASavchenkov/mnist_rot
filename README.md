# mnist_rot
A repository for experimentation regarding regression and classification
on mnist.

The rotate_mnist script takes the files in mnist_png, and (currently)
rotates each one randomly five times, appending the rotation in degrees to the label.

Notable current experiments involve:

1,) measuring the effectiveness of simply applying regression to the scaled angle input
  vs converting the single angle label into two labels, the cos and sin of the angle.
  HYPOTHETICALLY, the latter method should improve performance due to not being a very
  "irregular" and discontinuous function.
  
2,) attempting to learn classification and regression with only the final layer being
  separate. HYPOTHETICALLY the network would learn to segment previous layers into info
  that is useful for regression, and info that is useful for classification.
  follow up question, what level of sepaeration would yield optimal effectiveness?

all of the code is unfotunately still in a single file. When I have time I will split
boilerplate convnet code and interesting bits that change often into separate files.
