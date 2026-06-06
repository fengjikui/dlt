Heresy: issues in code that not only must be **avoided** but actively **detected and removed**.

## opus heresy
* opus talks to itself in docstrings instead of writing code comments in relevant places
* opus talks about some obscure downstream cases in upstream code (ie. base class is full of details of particular implementation or explains particular use case)
* opus invents obscure terminology and uses it everywhere. does not follow terminology used in docs
* opus writes extremely detailed and complex sentences to explain simple things

## arithmetic heresy
* decimals are converted to binary floats and back
* binary floats are used to construct decimals. ie. `Decimal(1.1)`

## datetime heresy
* naive datetimes are used without a good reason. the only exception are end user requirements

## terms heresy
* use of following words constitutes heresy: gate/gating, graft

## comments heresy
* block separator comments as below
```
#------------------------------
# comment
#------------------------------
```
or similar. actually any form