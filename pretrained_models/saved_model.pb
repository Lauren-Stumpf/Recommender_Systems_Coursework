¬
è
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02v2.7.0-0-gc256c071bb28Ôü

mlp_embedding_user/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	/ *.
shared_namemlp_embedding_user/embeddings

1mlp_embedding_user/embeddings/Read/ReadVariableOpReadVariableOpmlp_embedding_user/embeddings*
_output_shapes
:	/ *
dtype0

mlp_embedding_item/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ú *.
shared_namemlp_embedding_item/embeddings

1mlp_embedding_item/embeddings/Read/ReadVariableOpReadVariableOpmlp_embedding_item/embeddings*
_output_shapes
:	ú *
dtype0

mf_embedding_user/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	/*-
shared_namemf_embedding_user/embeddings

0mf_embedding_user/embeddings/Read/ReadVariableOpReadVariableOpmf_embedding_user/embeddings*
_output_shapes
:	/*
dtype0

mf_embedding_item/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ú*-
shared_namemf_embedding_item/embeddings

0mf_embedding_item/embeddings/Read/ReadVariableOpReadVariableOpmf_embedding_item/embeddings*
_output_shapes
:	ú*
dtype0
v
layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namelayer1/kernel
o
!layer1/kernel/Read/ReadVariableOpReadVariableOplayer1/kernel*
_output_shapes

:@ *
dtype0
n
layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelayer1/bias
g
layer1/bias/Read/ReadVariableOpReadVariableOplayer1/bias*
_output_shapes
: *
dtype0
v
layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namelayer2/kernel
o
!layer2/kernel/Read/ReadVariableOpReadVariableOplayer2/kernel*
_output_shapes

: *
dtype0
n
layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer2/bias
g
layer2/bias/Read/ReadVariableOpReadVariableOplayer2/bias*
_output_shapes
:*
dtype0
v
layer3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namelayer3/kernel
o
!layer3/kernel/Read/ReadVariableOpReadVariableOplayer3/kernel*
_output_shapes

:*
dtype0
n
layer3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer3/bias
g
layer3/bias/Read/ReadVariableOpReadVariableOplayer3/bias*
_output_shapes
:*
dtype0
~
prediction/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_nameprediction/kernel
w
%prediction/kernel/Read/ReadVariableOpReadVariableOpprediction/kernel*
_output_shapes

:*
dtype0
v
prediction/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameprediction/bias
o
#prediction/bias/Read/ReadVariableOpReadVariableOpprediction/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
¥
$Adam/mlp_embedding_user/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	/ *5
shared_name&$Adam/mlp_embedding_user/embeddings/m

8Adam/mlp_embedding_user/embeddings/m/Read/ReadVariableOpReadVariableOp$Adam/mlp_embedding_user/embeddings/m*
_output_shapes
:	/ *
dtype0
¥
$Adam/mlp_embedding_item/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ú *5
shared_name&$Adam/mlp_embedding_item/embeddings/m

8Adam/mlp_embedding_item/embeddings/m/Read/ReadVariableOpReadVariableOp$Adam/mlp_embedding_item/embeddings/m*
_output_shapes
:	ú *
dtype0
£
#Adam/mf_embedding_user/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	/*4
shared_name%#Adam/mf_embedding_user/embeddings/m

7Adam/mf_embedding_user/embeddings/m/Read/ReadVariableOpReadVariableOp#Adam/mf_embedding_user/embeddings/m*
_output_shapes
:	/*
dtype0
£
#Adam/mf_embedding_item/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ú*4
shared_name%#Adam/mf_embedding_item/embeddings/m

7Adam/mf_embedding_item/embeddings/m/Read/ReadVariableOpReadVariableOp#Adam/mf_embedding_item/embeddings/m*
_output_shapes
:	ú*
dtype0

Adam/layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *%
shared_nameAdam/layer1/kernel/m
}
(Adam/layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/m*
_output_shapes

:@ *
dtype0
|
Adam/layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/layer1/bias/m
u
&Adam/layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/m*
_output_shapes
: *
dtype0

Adam/layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameAdam/layer2/kernel/m
}
(Adam/layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/m*
_output_shapes

: *
dtype0
|
Adam/layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer2/bias/m
u
&Adam/layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/m*
_output_shapes
:*
dtype0

Adam/layer3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/layer3/kernel/m
}
(Adam/layer3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer3/kernel/m*
_output_shapes

:*
dtype0
|
Adam/layer3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer3/bias/m
u
&Adam/layer3/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer3/bias/m*
_output_shapes
:*
dtype0

Adam/prediction/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/prediction/kernel/m

,Adam/prediction/kernel/m/Read/ReadVariableOpReadVariableOpAdam/prediction/kernel/m*
_output_shapes

:*
dtype0

Adam/prediction/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/prediction/bias/m
}
*Adam/prediction/bias/m/Read/ReadVariableOpReadVariableOpAdam/prediction/bias/m*
_output_shapes
:*
dtype0
¥
$Adam/mlp_embedding_user/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	/ *5
shared_name&$Adam/mlp_embedding_user/embeddings/v

8Adam/mlp_embedding_user/embeddings/v/Read/ReadVariableOpReadVariableOp$Adam/mlp_embedding_user/embeddings/v*
_output_shapes
:	/ *
dtype0
¥
$Adam/mlp_embedding_item/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ú *5
shared_name&$Adam/mlp_embedding_item/embeddings/v

8Adam/mlp_embedding_item/embeddings/v/Read/ReadVariableOpReadVariableOp$Adam/mlp_embedding_item/embeddings/v*
_output_shapes
:	ú *
dtype0
£
#Adam/mf_embedding_user/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	/*4
shared_name%#Adam/mf_embedding_user/embeddings/v

7Adam/mf_embedding_user/embeddings/v/Read/ReadVariableOpReadVariableOp#Adam/mf_embedding_user/embeddings/v*
_output_shapes
:	/*
dtype0
£
#Adam/mf_embedding_item/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ú*4
shared_name%#Adam/mf_embedding_item/embeddings/v

7Adam/mf_embedding_item/embeddings/v/Read/ReadVariableOpReadVariableOp#Adam/mf_embedding_item/embeddings/v*
_output_shapes
:	ú*
dtype0

Adam/layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *%
shared_nameAdam/layer1/kernel/v
}
(Adam/layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer1/kernel/v*
_output_shapes

:@ *
dtype0
|
Adam/layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/layer1/bias/v
u
&Adam/layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer1/bias/v*
_output_shapes
: *
dtype0

Adam/layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *%
shared_nameAdam/layer2/kernel/v
}
(Adam/layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer2/kernel/v*
_output_shapes

: *
dtype0
|
Adam/layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer2/bias/v
u
&Adam/layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer2/bias/v*
_output_shapes
:*
dtype0

Adam/layer3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/layer3/kernel/v
}
(Adam/layer3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer3/kernel/v*
_output_shapes

:*
dtype0
|
Adam/layer3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/layer3/bias/v
u
&Adam/layer3/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer3/bias/v*
_output_shapes
:*
dtype0

Adam/prediction/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/prediction/kernel/v

,Adam/prediction/kernel/v/Read/ReadVariableOpReadVariableOpAdam/prediction/kernel/v*
_output_shapes

:*
dtype0

Adam/prediction/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/prediction/bias/v
}
*Adam/prediction/bias/v/Read/ReadVariableOpReadVariableOpAdam/prediction/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ÓU
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*U
valueUBU BúT
¥
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
 
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
b

embeddings
	variables
trainable_variables
 regularization_losses
!	keras_api
R
"	variables
#trainable_variables
$regularization_losses
%	keras_api
R
&	variables
'trainable_variables
(regularization_losses
)	keras_api
R
*	variables
+trainable_variables
,regularization_losses
-	keras_api
b
.
embeddings
/	variables
0trainable_variables
1regularization_losses
2	keras_api
b
3
embeddings
4	variables
5trainable_variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
R
>	variables
?trainable_variables
@regularization_losses
A	keras_api
R
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
h

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
R
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
h

Pkernel
Qbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
R
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
h

Zkernel
[bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
°
`iter

abeta_1

bbeta_2
	cdecay
dlearning_ratemºm».m¼3m½8m¾9m¿FmÀGmÁPmÂQmÃZmÄ[mÅvÆvÇ.vÈ3vÉ8vÊ9vËFvÌGvÍPvÎQvÏZvÐ[vÑ
V
0
1
.2
33
84
95
F6
G7
P8
Q9
Z10
[11
V
0
1
.2
33
84
95
F6
G7
P8
Q9
Z10
[11
 
­
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
 
mk
VARIABLE_VALUEmlp_embedding_user/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
mk
VARIABLE_VALUEmlp_embedding_item/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
 regularization_losses
 
 
 
­
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
"	variables
#trainable_variables
$regularization_losses
 
 
 
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
&	variables
'trainable_variables
(regularization_losses
 
 
 
°
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
lj
VARIABLE_VALUEmf_embedding_user/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE

.0

.0
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
lj
VARIABLE_VALUEmf_embedding_item/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE

30

30
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
4	variables
5trainable_variables
6regularization_losses
YW
VARIABLE_VALUElayer1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
:	variables
;trainable_variables
<regularization_losses
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
YW
VARIABLE_VALUElayer2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1

F0
G1
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
 
 
 
²
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
YW
VARIABLE_VALUElayer3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElayer3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1

P0
Q1
 
²
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
 
 
 
²
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
][
VARIABLE_VALUEprediction/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEprediction/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1

Z0
[1
 
²
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
\	variables
]trainable_variables
^regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16

µ0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

¶total

·count
¸	variables
¹	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

¶0
·1

¸	variables

VARIABLE_VALUE$Adam/mlp_embedding_user/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/mlp_embedding_item/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/mf_embedding_user/embeddings/mVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/mf_embedding_item/embeddings/mVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/prediction/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/prediction/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/mlp_embedding_user/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adam/mlp_embedding_item/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/mf_embedding_user/embeddings/vVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/mf_embedding_item/embeddings/vVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/layer3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/layer3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/prediction/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/prediction/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_args_0Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
{
serving_default_args_0_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ù
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_1mlp_embedding_item/embeddingsmlp_embedding_user/embeddingslayer1/kernellayer1/biasmf_embedding_item/embeddingsmf_embedding_user/embeddingslayer2/kernellayer2/biaslayer3/kernellayer3/biasprediction/kernelprediction/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *0
f+R)
'__inference_signature_wrapper_167840584
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ö
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename1mlp_embedding_user/embeddings/Read/ReadVariableOp1mlp_embedding_item/embeddings/Read/ReadVariableOp0mf_embedding_user/embeddings/Read/ReadVariableOp0mf_embedding_item/embeddings/Read/ReadVariableOp!layer1/kernel/Read/ReadVariableOplayer1/bias/Read/ReadVariableOp!layer2/kernel/Read/ReadVariableOplayer2/bias/Read/ReadVariableOp!layer3/kernel/Read/ReadVariableOplayer3/bias/Read/ReadVariableOp%prediction/kernel/Read/ReadVariableOp#prediction/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp8Adam/mlp_embedding_user/embeddings/m/Read/ReadVariableOp8Adam/mlp_embedding_item/embeddings/m/Read/ReadVariableOp7Adam/mf_embedding_user/embeddings/m/Read/ReadVariableOp7Adam/mf_embedding_item/embeddings/m/Read/ReadVariableOp(Adam/layer1/kernel/m/Read/ReadVariableOp&Adam/layer1/bias/m/Read/ReadVariableOp(Adam/layer2/kernel/m/Read/ReadVariableOp&Adam/layer2/bias/m/Read/ReadVariableOp(Adam/layer3/kernel/m/Read/ReadVariableOp&Adam/layer3/bias/m/Read/ReadVariableOp,Adam/prediction/kernel/m/Read/ReadVariableOp*Adam/prediction/bias/m/Read/ReadVariableOp8Adam/mlp_embedding_user/embeddings/v/Read/ReadVariableOp8Adam/mlp_embedding_item/embeddings/v/Read/ReadVariableOp7Adam/mf_embedding_user/embeddings/v/Read/ReadVariableOp7Adam/mf_embedding_item/embeddings/v/Read/ReadVariableOp(Adam/layer1/kernel/v/Read/ReadVariableOp&Adam/layer1/bias/v/Read/ReadVariableOp(Adam/layer2/kernel/v/Read/ReadVariableOp&Adam/layer2/bias/v/Read/ReadVariableOp(Adam/layer3/kernel/v/Read/ReadVariableOp&Adam/layer3/bias/v/Read/ReadVariableOp,Adam/prediction/kernel/v/Read/ReadVariableOp*Adam/prediction/bias/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_save_167841400


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemlp_embedding_user/embeddingsmlp_embedding_item/embeddingsmf_embedding_user/embeddingsmf_embedding_item/embeddingslayer1/kernellayer1/biaslayer2/kernellayer2/biaslayer3/kernellayer3/biasprediction/kernelprediction/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount$Adam/mlp_embedding_user/embeddings/m$Adam/mlp_embedding_item/embeddings/m#Adam/mf_embedding_user/embeddings/m#Adam/mf_embedding_item/embeddings/mAdam/layer1/kernel/mAdam/layer1/bias/mAdam/layer2/kernel/mAdam/layer2/bias/mAdam/layer3/kernel/mAdam/layer3/bias/mAdam/prediction/kernel/mAdam/prediction/bias/m$Adam/mlp_embedding_user/embeddings/v$Adam/mlp_embedding_item/embeddings/v#Adam/mf_embedding_user/embeddings/v#Adam/mf_embedding_item/embeddings/vAdam/layer1/kernel/vAdam/layer1/bias/vAdam/layer2/kernel/vAdam/layer2/bias/vAdam/layer3/kernel/vAdam/layer3/bias/vAdam/prediction/kernel/vAdam/prediction/bias/v*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference__traced_restore_167841539ð¬
Å
x
L__inference_concatenate_1_layer_call_and_return_conditional_losses_167841150
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
À
d
H__inference_flatten_3_layer_call_and_return_conditional_losses_167840033

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ëT
Å

$__inference__wrapped_model_167839972

args_0
args_0_1F
3model_mlp_embedding_item_embedding_lookup_167839910:	ú F
3model_mlp_embedding_user_embedding_lookup_167839915:	/ =
+model_layer1_matmul_readvariableop_resource:@ :
,model_layer1_biasadd_readvariableop_resource: E
2model_mf_embedding_item_embedding_lookup_167839933:	úE
2model_mf_embedding_user_embedding_lookup_167839938:	/=
+model_layer2_matmul_readvariableop_resource: :
,model_layer2_biasadd_readvariableop_resource:=
+model_layer3_matmul_readvariableop_resource::
,model_layer3_biasadd_readvariableop_resource:A
/model_prediction_matmul_readvariableop_resource:>
0model_prediction_biasadd_readvariableop_resource:
identity¢#model/layer1/BiasAdd/ReadVariableOp¢"model/layer1/MatMul/ReadVariableOp¢#model/layer2/BiasAdd/ReadVariableOp¢"model/layer2/MatMul/ReadVariableOp¢#model/layer3/BiasAdd/ReadVariableOp¢"model/layer3/MatMul/ReadVariableOp¢(model/mf_embedding_item/embedding_lookup¢(model/mf_embedding_user/embedding_lookup¢)model/mlp_embedding_item/embedding_lookup¢)model/mlp_embedding_user/embedding_lookup¢'model/prediction/BiasAdd/ReadVariableOp¢&model/prediction/MatMul/ReadVariableOp
)model/mlp_embedding_item/embedding_lookupResourceGather3model_mlp_embedding_item_embedding_lookup_167839910args_0_1*
Tindices0*F
_class<
:8loc:@model/mlp_embedding_item/embedding_lookup/167839910*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0ð
2model/mlp_embedding_item/embedding_lookup/IdentityIdentity2model/mlp_embedding_item/embedding_lookup:output:0*
T0*F
_class<
:8loc:@model/mlp_embedding_item/embedding_lookup/167839910*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
4model/mlp_embedding_item/embedding_lookup/Identity_1Identity;model/mlp_embedding_item/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)model/mlp_embedding_user/embedding_lookupResourceGather3model_mlp_embedding_user_embedding_lookup_167839915args_0*
Tindices0*F
_class<
:8loc:@model/mlp_embedding_user/embedding_lookup/167839915*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0ð
2model/mlp_embedding_user/embedding_lookup/IdentityIdentity2model/mlp_embedding_user/embedding_lookup:output:0*
T0*F
_class<
:8loc:@model/mlp_embedding_user/embedding_lookup/167839915*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ³
4model/mlp_embedding_user/embedding_lookup/Identity_1Identity;model/mlp_embedding_user/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
model/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ³
model/flatten_2/ReshapeReshape=model/mlp_embedding_user/embedding_lookup/Identity_1:output:0model/flatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
model/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ³
model/flatten_3/ReshapeReshape=model/mlp_embedding_item/embedding_lookup/Identity_1:output:0model/flatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ _
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ë
model/concatenate/concatConcatV2 model/flatten_2/Reshape:output:0 model/flatten_3/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
"model/layer1/MatMul/ReadVariableOpReadVariableOp+model_layer1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
model/layer1/MatMulMatMul!model/concatenate/concat:output:0*model/layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
#model/layer1/BiasAdd/ReadVariableOpReadVariableOp,model_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
model/layer1/BiasAddBiasAddmodel/layer1/MatMul:product:0+model/layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
model/layer1/ReluRelumodel/layer1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(model/mf_embedding_item/embedding_lookupResourceGather2model_mf_embedding_item_embedding_lookup_167839933args_0_1*
Tindices0*E
_class;
97loc:@model/mf_embedding_item/embedding_lookup/167839933*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0í
1model/mf_embedding_item/embedding_lookup/IdentityIdentity1model/mf_embedding_item/embedding_lookup:output:0*
T0*E
_class;
97loc:@model/mf_embedding_item/embedding_lookup/167839933*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
3model/mf_embedding_item/embedding_lookup/Identity_1Identity:model/mf_embedding_item/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(model/mf_embedding_user/embedding_lookupResourceGather2model_mf_embedding_user_embedding_lookup_167839938args_0*
Tindices0*E
_class;
97loc:@model/mf_embedding_user/embedding_lookup/167839938*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0í
1model/mf_embedding_user/embedding_lookup/IdentityIdentity1model/mf_embedding_user/embedding_lookup:output:0*
T0*E
_class;
97loc:@model/mf_embedding_user/embedding_lookup/167839938*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
3model/mf_embedding_user/embedding_lookup/Identity_1Identity:model/mf_embedding_user/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model/layer2/MatMul/ReadVariableOpReadVariableOp+model_layer2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
model/layer2/MatMulMatMulmodel/layer1/Relu:activations:0*model/layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model/layer2/BiasAdd/ReadVariableOpReadVariableOp,model_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
model/layer2/BiasAddBiasAddmodel/layer2/MatMul:product:0+model/layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
model/layer2/ReluRelumodel/layer2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ²
model/flatten_1/ReshapeReshape<model/mf_embedding_user/embedding_lookup/Identity_1:output:0model/flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ®
model/flatten/ReshapeReshape<model/mf_embedding_item/embedding_lookup/Identity_1:output:0model/flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/multiply/mulMul model/flatten_1/Reshape:output:0model/flatten/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model/layer3/MatMul/ReadVariableOpReadVariableOp+model_layer3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
model/layer3/MatMulMatMulmodel/layer2/Relu:activations:0*model/layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model/layer3/BiasAdd/ReadVariableOpReadVariableOp,model_layer3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
model/layer3/BiasAddBiasAddmodel/layer3/MatMul:product:0+model/layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
model/layer3/ReluRelumodel/layer3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ä
model/concatenate_1/concatConcatV2model/multiply/mul:z:0model/layer3/Relu:activations:0(model/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/prediction/MatMul/ReadVariableOpReadVariableOp/model_prediction_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¨
model/prediction/MatMulMatMul#model/concatenate_1/concat:output:0.model/prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model/prediction/BiasAdd/ReadVariableOpReadVariableOp0model_prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
model/prediction/BiasAddBiasAdd!model/prediction/MatMul:product:0/model/prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
model/prediction/SigmoidSigmoid!model/prediction/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
IdentityIdentitymodel/prediction/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp$^model/layer1/BiasAdd/ReadVariableOp#^model/layer1/MatMul/ReadVariableOp$^model/layer2/BiasAdd/ReadVariableOp#^model/layer2/MatMul/ReadVariableOp$^model/layer3/BiasAdd/ReadVariableOp#^model/layer3/MatMul/ReadVariableOp)^model/mf_embedding_item/embedding_lookup)^model/mf_embedding_user/embedding_lookup*^model/mlp_embedding_item/embedding_lookup*^model/mlp_embedding_user/embedding_lookup(^model/prediction/BiasAdd/ReadVariableOp'^model/prediction/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2J
#model/layer1/BiasAdd/ReadVariableOp#model/layer1/BiasAdd/ReadVariableOp2H
"model/layer1/MatMul/ReadVariableOp"model/layer1/MatMul/ReadVariableOp2J
#model/layer2/BiasAdd/ReadVariableOp#model/layer2/BiasAdd/ReadVariableOp2H
"model/layer2/MatMul/ReadVariableOp"model/layer2/MatMul/ReadVariableOp2J
#model/layer3/BiasAdd/ReadVariableOp#model/layer3/BiasAdd/ReadVariableOp2H
"model/layer3/MatMul/ReadVariableOp"model/layer3/MatMul/ReadVariableOp2T
(model/mf_embedding_item/embedding_lookup(model/mf_embedding_item/embedding_lookup2T
(model/mf_embedding_user/embedding_lookup(model/mf_embedding_user/embedding_lookup2V
)model/mlp_embedding_item/embedding_lookup)model/mlp_embedding_item/embedding_lookup2V
)model/mlp_embedding_user/embedding_lookup)model/mlp_embedding_user/embedding_lookup2R
'model/prediction/BiasAdd/ReadVariableOp'model/prediction/BiasAdd/ReadVariableOp2P
&model/prediction/MatMul/ReadVariableOp&model/prediction/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
³
ò
P__inference_mf_embedding_user_layer_call_and_return_conditional_losses_167840979

inputs-
embedding_lookup_167840967:	/
identity¢embedding_lookup¢>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp¿
embedding_lookupResourceGatherembedding_lookup_167840967inputs*
Tindices0*-
_class#
!loc:@embedding_lookup/167840967*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0¥
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/167840967*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_167840967*
_output_shapes
:	/*
dtype0«
/mf_embedding_user/embeddings/Regularizer/SquareSquareFmf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_user/embeddings/Regularizer/SumSum3mf_embedding_user/embeddings/Regularizer/Square:y:07mf_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_user/embeddings/Regularizer/mulMul7mf_embedding_user/embeddings/Regularizer/mul/x:output:05mf_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^embedding_lookup?^mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup2
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
ò
P__inference_mf_embedding_user_layer_call_and_return_conditional_losses_167840101

inputs-
embedding_lookup_167840089:	/
identity¢embedding_lookup¢>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp¿
embedding_lookupResourceGatherembedding_lookup_167840089inputs*
Tindices0*-
_class#
!loc:@embedding_lookup/167840089*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0¥
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/167840089*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_167840089*
_output_shapes
:	/*
dtype0«
/mf_embedding_user/embeddings/Regularizer/SquareSquareFmf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_user/embeddings/Regularizer/SumSum3mf_embedding_user/embeddings/Regularizer/Square:y:07mf_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_user/embeddings/Regularizer/mulMul7mf_embedding_user/embeddings/Regularizer/mul/x:output:05mf_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^embedding_lookup?^mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup2
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
X
,__inference_multiply_layer_call_fn_167841099
inputs_0
inputs_1
identity¿
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_multiply_layer_call_and_return_conditional_losses_167840150`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ä

*__inference_layer3_layer_call_fn_167841120

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_layer3_layer_call_and_return_conditional_losses_167840169o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
ò
P__inference_mf_embedding_item_layer_call_and_return_conditional_losses_167841007

inputs-
embedding_lookup_167840995:	ú
identity¢embedding_lookup¢>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp¿
embedding_lookupResourceGatherembedding_lookup_167840995inputs*
Tindices0*-
_class#
!loc:@embedding_lookup/167840995*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0¥
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/167840995*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_167840995*
_output_shapes
:	ú*
dtype0«
/mf_embedding_item/embeddings/Regularizer/SquareSquareFmf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_item/embeddings/Regularizer/SumSum3mf_embedding_item/embeddings/Regularizer/Square:y:07mf_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_item/embeddings/Regularizer/mulMul7mf_embedding_item/embeddings/Regularizer/mul/x:output:05mf_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^embedding_lookup?^mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup2
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»[
õ
"__inference__traced_save_167841400
file_prefix<
8savev2_mlp_embedding_user_embeddings_read_readvariableop<
8savev2_mlp_embedding_item_embeddings_read_readvariableop;
7savev2_mf_embedding_user_embeddings_read_readvariableop;
7savev2_mf_embedding_item_embeddings_read_readvariableop,
(savev2_layer1_kernel_read_readvariableop*
&savev2_layer1_bias_read_readvariableop,
(savev2_layer2_kernel_read_readvariableop*
&savev2_layer2_bias_read_readvariableop,
(savev2_layer3_kernel_read_readvariableop*
&savev2_layer3_bias_read_readvariableop0
,savev2_prediction_kernel_read_readvariableop.
*savev2_prediction_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopC
?savev2_adam_mlp_embedding_user_embeddings_m_read_readvariableopC
?savev2_adam_mlp_embedding_item_embeddings_m_read_readvariableopB
>savev2_adam_mf_embedding_user_embeddings_m_read_readvariableopB
>savev2_adam_mf_embedding_item_embeddings_m_read_readvariableop3
/savev2_adam_layer1_kernel_m_read_readvariableop1
-savev2_adam_layer1_bias_m_read_readvariableop3
/savev2_adam_layer2_kernel_m_read_readvariableop1
-savev2_adam_layer2_bias_m_read_readvariableop3
/savev2_adam_layer3_kernel_m_read_readvariableop1
-savev2_adam_layer3_bias_m_read_readvariableop7
3savev2_adam_prediction_kernel_m_read_readvariableop5
1savev2_adam_prediction_bias_m_read_readvariableopC
?savev2_adam_mlp_embedding_user_embeddings_v_read_readvariableopC
?savev2_adam_mlp_embedding_item_embeddings_v_read_readvariableopB
>savev2_adam_mf_embedding_user_embeddings_v_read_readvariableopB
>savev2_adam_mf_embedding_item_embeddings_v_read_readvariableop3
/savev2_adam_layer1_kernel_v_read_readvariableop1
-savev2_adam_layer1_bias_v_read_readvariableop3
/savev2_adam_layer2_kernel_v_read_readvariableop1
-savev2_adam_layer2_bias_v_read_readvariableop3
/savev2_adam_layer3_kernel_v_read_readvariableop1
-savev2_adam_layer3_bias_v_read_readvariableop7
3savev2_adam_prediction_kernel_v_read_readvariableop5
1savev2_adam_prediction_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ó
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*
valueB,B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÅ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B µ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:08savev2_mlp_embedding_user_embeddings_read_readvariableop8savev2_mlp_embedding_item_embeddings_read_readvariableop7savev2_mf_embedding_user_embeddings_read_readvariableop7savev2_mf_embedding_item_embeddings_read_readvariableop(savev2_layer1_kernel_read_readvariableop&savev2_layer1_bias_read_readvariableop(savev2_layer2_kernel_read_readvariableop&savev2_layer2_bias_read_readvariableop(savev2_layer3_kernel_read_readvariableop&savev2_layer3_bias_read_readvariableop,savev2_prediction_kernel_read_readvariableop*savev2_prediction_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop?savev2_adam_mlp_embedding_user_embeddings_m_read_readvariableop?savev2_adam_mlp_embedding_item_embeddings_m_read_readvariableop>savev2_adam_mf_embedding_user_embeddings_m_read_readvariableop>savev2_adam_mf_embedding_item_embeddings_m_read_readvariableop/savev2_adam_layer1_kernel_m_read_readvariableop-savev2_adam_layer1_bias_m_read_readvariableop/savev2_adam_layer2_kernel_m_read_readvariableop-savev2_adam_layer2_bias_m_read_readvariableop/savev2_adam_layer3_kernel_m_read_readvariableop-savev2_adam_layer3_bias_m_read_readvariableop3savev2_adam_prediction_kernel_m_read_readvariableop1savev2_adam_prediction_bias_m_read_readvariableop?savev2_adam_mlp_embedding_user_embeddings_v_read_readvariableop?savev2_adam_mlp_embedding_item_embeddings_v_read_readvariableop>savev2_adam_mf_embedding_user_embeddings_v_read_readvariableop>savev2_adam_mf_embedding_item_embeddings_v_read_readvariableop/savev2_adam_layer1_kernel_v_read_readvariableop-savev2_adam_layer1_bias_v_read_readvariableop/savev2_adam_layer2_kernel_v_read_readvariableop-savev2_adam_layer2_bias_v_read_readvariableop/savev2_adam_layer3_kernel_v_read_readvariableop-savev2_adam_layer3_bias_v_read_readvariableop3savev2_adam_prediction_kernel_v_read_readvariableop1savev2_adam_prediction_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ë
_input_shapesÙ
Ö: :	/ :	ú :	/:	ú:@ : : :::::: : : : : : : :	/ :	ú :	/:	ú:@ : : ::::::	/ :	ú :	/:	ú:@ : : :::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	/ :%!

_output_shapes
:	ú :%!

_output_shapes
:	/:%!

_output_shapes
:	ú:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	/ :%!

_output_shapes
:	ú :%!

_output_shapes
:	/:%!

_output_shapes
:	ú:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::% !

_output_shapes
:	/ :%!!

_output_shapes
:	ú :%"!

_output_shapes
:	/:%#!

_output_shapes
:	ú:$$ 

_output_shapes

:@ : %

_output_shapes
: :$& 

_output_shapes

: : '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
::$* 

_output_shapes

:: +

_output_shapes
::,

_output_shapes
: 
­
I
-__inference_flatten_3_layer_call_fn_167840932

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_3_layer_call_and_return_conditional_losses_167840033`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¨
E__inference_layer3_layer_call_and_return_conditional_losses_167841137

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/layer3/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
 layer3/kernel/Regularizer/SquareSquare7layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer3/kernel/Regularizer/SumSum$layer3/kernel/Regularizer/Square:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^layer3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/layer3/kernel/Regularizer/Square/ReadVariableOp/layer3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
b
F__inference_flatten_layer_call_and_return_conditional_losses_167841061

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¨
E__inference_layer1_layer_call_and_return_conditional_losses_167840061

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/layer1/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@ p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
½

6__inference_mlp_embedding_user_layer_call_fn_167840873

inputs
unknown:	/ 
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_167840015s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
ò
P__inference_mf_embedding_item_layer_call_and_return_conditional_losses_167840082

inputs-
embedding_lookup_167840070:	ú
identity¢embedding_lookup¢>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp¿
embedding_lookupResourceGatherembedding_lookup_167840070inputs*
Tindices0*-
_class#
!loc:@embedding_lookup/167840070*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0¥
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/167840070*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_167840070*
_output_shapes
:	ú*
dtype0«
/mf_embedding_item/embeddings/Regularizer/SquareSquareFmf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_item/embeddings/Regularizer/SumSum3mf_embedding_item/embeddings/Regularizer/Square:y:07mf_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_item/embeddings/Regularizer/mulMul7mf_embedding_item/embeddings/Regularizer/mul/x:output:05mf_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^embedding_lookup?^mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup2
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
d
H__inference_flatten_1_layer_call_and_return_conditional_losses_167841050

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¨
E__inference_layer3_layer_call_and_return_conditional_losses_167840169

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/layer3/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0
 layer3/kernel/Regularizer/SquareSquare7layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer3/kernel/Regularizer/SumSum$layer3/kernel/Regularizer/Square:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^layer3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/layer3/kernel/Regularizer/Square/ReadVariableOp/layer3/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
]
1__inference_concatenate_1_layer_call_fn_167841143
inputs_0
inputs_1
identityÄ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_concatenate_1_layer_call_and_return_conditional_losses_167840182`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1


ú
I__inference_prediction_layer_call_and_return_conditional_losses_167841170

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

.__inference_prediction_layer_call_fn_167841159

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_prediction_layer_call_and_return_conditional_losses_167840195o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
ó
D__inference_model_layer_call_and_return_conditional_losses_167840752
inputs_0
inputs_1@
-mlp_embedding_item_embedding_lookup_167840648:	ú @
-mlp_embedding_user_embedding_lookup_167840653:	/ 7
%layer1_matmul_readvariableop_resource:@ 4
&layer1_biasadd_readvariableop_resource: ?
,mf_embedding_item_embedding_lookup_167840671:	ú?
,mf_embedding_user_embedding_lookup_167840676:	/7
%layer2_matmul_readvariableop_resource: 4
&layer2_biasadd_readvariableop_resource:7
%layer3_matmul_readvariableop_resource:4
&layer3_biasadd_readvariableop_resource:;
)prediction_matmul_readvariableop_resource:8
*prediction_biasadd_readvariableop_resource:
identity¢layer1/BiasAdd/ReadVariableOp¢layer1/MatMul/ReadVariableOp¢/layer1/kernel/Regularizer/Square/ReadVariableOp¢layer2/BiasAdd/ReadVariableOp¢layer2/MatMul/ReadVariableOp¢/layer2/kernel/Regularizer/Square/ReadVariableOp¢layer3/BiasAdd/ReadVariableOp¢layer3/MatMul/ReadVariableOp¢/layer3/kernel/Regularizer/Square/ReadVariableOp¢"mf_embedding_item/embedding_lookup¢>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp¢"mf_embedding_user/embedding_lookup¢>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp¢#mlp_embedding_item/embedding_lookup¢?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp¢#mlp_embedding_user/embedding_lookup¢?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp¢!prediction/BiasAdd/ReadVariableOp¢ prediction/MatMul/ReadVariableOpú
#mlp_embedding_item/embedding_lookupResourceGather-mlp_embedding_item_embedding_lookup_167840648inputs_1*
Tindices0*@
_class6
42loc:@mlp_embedding_item/embedding_lookup/167840648*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0Þ
,mlp_embedding_item/embedding_lookup/IdentityIdentity,mlp_embedding_item/embedding_lookup:output:0*
T0*@
_class6
42loc:@mlp_embedding_item/embedding_lookup/167840648*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §
.mlp_embedding_item/embedding_lookup/Identity_1Identity5mlp_embedding_item/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ú
#mlp_embedding_user/embedding_lookupResourceGather-mlp_embedding_user_embedding_lookup_167840653inputs_0*
Tindices0*@
_class6
42loc:@mlp_embedding_user/embedding_lookup/167840653*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0Þ
,mlp_embedding_user/embedding_lookup/IdentityIdentity,mlp_embedding_user/embedding_lookup:output:0*
T0*@
_class6
42loc:@mlp_embedding_user/embedding_lookup/167840653*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §
.mlp_embedding_user/embedding_lookup/Identity_1Identity5mlp_embedding_user/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¡
flatten_2/ReshapeReshape7mlp_embedding_user/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¡
flatten_3/ReshapeReshape7mlp_embedding_item/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :³
concatenate/concatConcatV2flatten_2/Reshape:output:0flatten_3/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
layer1/MatMulMatMulconcatenate/concat:output:0$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ÷
"mf_embedding_item/embedding_lookupResourceGather,mf_embedding_item_embedding_lookup_167840671inputs_1*
Tindices0*?
_class5
31loc:@mf_embedding_item/embedding_lookup/167840671*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Û
+mf_embedding_item/embedding_lookup/IdentityIdentity+mf_embedding_item/embedding_lookup:output:0*
T0*?
_class5
31loc:@mf_embedding_item/embedding_lookup/167840671*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-mf_embedding_item/embedding_lookup/Identity_1Identity4mf_embedding_item/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ÷
"mf_embedding_user/embedding_lookupResourceGather,mf_embedding_user_embedding_lookup_167840676inputs_0*
Tindices0*?
_class5
31loc:@mf_embedding_user/embedding_lookup/167840676*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Û
+mf_embedding_user/embedding_lookup/IdentityIdentity+mf_embedding_user/embedding_lookup:output:0*
T0*?
_class5
31loc:@mf_embedding_user/embedding_lookup/167840676*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-mf_embedding_user/embedding_lookup/Identity_1Identity4mf_embedding_user/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
flatten_1/ReshapeReshape6mf_embedding_user/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten/ReshapeReshape6mf_embedding_item/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
multiply/mulMulflatten_1/Reshape:output:0flatten/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¬
concatenate_1/concatConcatV2multiply/mul:z:0layer3/Relu:activations:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 prediction/MatMul/ReadVariableOpReadVariableOp)prediction_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
prediction/MatMulMatMulconcatenate_1/concat:output:0(prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!prediction/BiasAdd/ReadVariableOpReadVariableOp*prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
prediction/BiasAddBiasAddprediction/MatMul:product:0)prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
prediction/SigmoidSigmoidprediction/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp-mlp_embedding_user_embedding_lookup_167840653*
_output_shapes
:	/ *
dtype0­
0mlp_embedding_user/embeddings/Regularizer/SquareSquareGmlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/ 
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_user/embeddings/Regularizer/SumSum4mlp_embedding_user/embeddings/Regularizer/Square:y:08mlp_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_user/embeddings/Regularizer/mulMul8mlp_embedding_user/embeddings/Regularizer/mul/x:output:06mlp_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: ®
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp-mlp_embedding_item_embedding_lookup_167840648*
_output_shapes
:	ú *
dtype0­
0mlp_embedding_item/embeddings/Regularizer/SquareSquareGmlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú 
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_item/embeddings/Regularizer/SumSum4mlp_embedding_item/embeddings/Regularizer/Square:y:08mlp_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_item/embeddings/Regularizer/mulMul8mlp_embedding_item/embeddings/Regularizer/mul/x:output:06mlp_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¬
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp,mf_embedding_user_embedding_lookup_167840676*
_output_shapes
:	/*
dtype0«
/mf_embedding_user/embeddings/Regularizer/SquareSquareFmf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_user/embeddings/Regularizer/SumSum3mf_embedding_user/embeddings/Regularizer/Square:y:07mf_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_user/embeddings/Regularizer/mulMul7mf_embedding_user/embeddings/Regularizer/mul/x:output:05mf_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¬
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp,mf_embedding_item_embedding_lookup_167840671*
_output_shapes
:	ú*
dtype0«
/mf_embedding_item/embeddings/Regularizer/SquareSquareFmf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_item/embeddings/Regularizer/SumSum3mf_embedding_item/embeddings/Regularizer/Square:y:07mf_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_item/embeddings/Regularizer/mulMul7mf_embedding_item/embeddings/Regularizer/mul/x:output:05mf_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@ p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
 layer3/kernel/Regularizer/SquareSquare7layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer3/kernel/Regularizer/SumSum$layer3/kernel/Regularizer/Square:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentityprediction/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
NoOpNoOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp0^layer2/kernel/Regularizer/Square/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp0^layer3/kernel/Regularizer/Square/ReadVariableOp#^mf_embedding_item/embedding_lookup?^mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp#^mf_embedding_user/embedding_lookup?^mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp$^mlp_embedding_item/embedding_lookup@^mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp$^mlp_embedding_user/embedding_lookup@^mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp"^prediction/BiasAdd/ReadVariableOp!^prediction/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp2b
/layer3/kernel/Regularizer/Square/ReadVariableOp/layer3/kernel/Regularizer/Square/ReadVariableOp2H
"mf_embedding_item/embedding_lookup"mf_embedding_item/embedding_lookup2
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp2H
"mf_embedding_user/embedding_lookup"mf_embedding_user/embedding_lookup2
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp2J
#mlp_embedding_item/embedding_lookup#mlp_embedding_item/embedding_lookup2
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp2J
#mlp_embedding_user/embedding_lookup#mlp_embedding_user/embedding_lookup2
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp2F
!prediction/BiasAdd/ReadVariableOp!prediction/BiasAdd/ReadVariableOp2D
 prediction/MatMul/ReadVariableOp prediction/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ä

*__inference_layer2_layer_call_fn_167841076

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_layer2_layer_call_and_return_conditional_losses_167840122o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
~
¡

D__inference_model_layer_call_and_return_conditional_losses_167840244

inputs
inputs_1/
mlp_embedding_item_167839997:	ú /
mlp_embedding_user_167840016:	/ "
layer1_167840062:@ 
layer1_167840064: .
mf_embedding_item_167840083:	ú.
mf_embedding_user_167840102:	/"
layer2_167840123: 
layer2_167840125:"
layer3_167840170:
layer3_167840172:&
prediction_167840196:"
prediction_167840198:
identity¢layer1/StatefulPartitionedCall¢/layer1/kernel/Regularizer/Square/ReadVariableOp¢layer2/StatefulPartitionedCall¢/layer2/kernel/Regularizer/Square/ReadVariableOp¢layer3/StatefulPartitionedCall¢/layer3/kernel/Regularizer/Square/ReadVariableOp¢)mf_embedding_item/StatefulPartitionedCall¢>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp¢)mf_embedding_user/StatefulPartitionedCall¢>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp¢*mlp_embedding_item/StatefulPartitionedCall¢?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp¢*mlp_embedding_user/StatefulPartitionedCall¢?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp¢"prediction/StatefulPartitionedCall
*mlp_embedding_item/StatefulPartitionedCallStatefulPartitionedCallinputs_1mlp_embedding_item_167839997*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_167839996
*mlp_embedding_user/StatefulPartitionedCallStatefulPartitionedCallinputsmlp_embedding_user_167840016*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_167840015ê
flatten_2/PartitionedCallPartitionedCall3mlp_embedding_user/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_2_layer_call_and_return_conditional_losses_167840025ê
flatten_3/PartitionedCallPartitionedCall3mlp_embedding_item/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_3_layer_call_and_return_conditional_losses_167840033
concatenate/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_concatenate_layer_call_and_return_conditional_losses_167840042
layer1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0layer1_167840062layer1_167840064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_layer1_layer_call_and_return_conditional_losses_167840061
)mf_embedding_item/StatefulPartitionedCallStatefulPartitionedCallinputs_1mf_embedding_item_167840083*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_mf_embedding_item_layer_call_and_return_conditional_losses_167840082
)mf_embedding_user/StatefulPartitionedCallStatefulPartitionedCallinputsmf_embedding_user_167840102*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_mf_embedding_user_layer_call_and_return_conditional_losses_167840101
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_167840123layer2_167840125*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_layer2_layer_call_and_return_conditional_losses_167840122é
flatten_1/PartitionedCallPartitionedCall2mf_embedding_user/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_167840134å
flatten/PartitionedCallPartitionedCall2mf_embedding_item/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_167840142ú
multiply/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_multiply_layer_call_and_return_conditional_losses_167840150
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_167840170layer3_167840172*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_layer3_layer_call_and_return_conditional_losses_167840169
concatenate_1/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_concatenate_1_layer_call_and_return_conditional_losses_167840182¡
"prediction/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0prediction_167840196prediction_167840198*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_prediction_layer_call_and_return_conditional_losses_167840195
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmlp_embedding_user_167840016*
_output_shapes
:	/ *
dtype0­
0mlp_embedding_user/embeddings/Regularizer/SquareSquareGmlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/ 
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_user/embeddings/Regularizer/SumSum4mlp_embedding_user/embeddings/Regularizer/Square:y:08mlp_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_user/embeddings/Regularizer/mulMul8mlp_embedding_user/embeddings/Regularizer/mul/x:output:06mlp_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmlp_embedding_item_167839997*
_output_shapes
:	ú *
dtype0­
0mlp_embedding_item/embeddings/Regularizer/SquareSquareGmlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú 
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_item/embeddings/Regularizer/SumSum4mlp_embedding_item/embeddings/Regularizer/Square:y:08mlp_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_item/embeddings/Regularizer/mulMul8mlp_embedding_item/embeddings/Regularizer/mul/x:output:06mlp_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmf_embedding_user_167840102*
_output_shapes
:	/*
dtype0«
/mf_embedding_user/embeddings/Regularizer/SquareSquareFmf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_user/embeddings/Regularizer/SumSum3mf_embedding_user/embeddings/Regularizer/Square:y:07mf_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_user/embeddings/Regularizer/mulMul7mf_embedding_user/embeddings/Regularizer/mul/x:output:05mf_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmf_embedding_item_167840083*
_output_shapes
:	ú*
dtype0«
/mf_embedding_item/embeddings/Regularizer/SquareSquareFmf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_item/embeddings/Regularizer/SumSum3mf_embedding_item/embeddings/Regularizer/Square:y:07mf_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_item/embeddings/Regularizer/mulMul7mf_embedding_item/embeddings/Regularizer/mul/x:output:05mf_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer1_167840062*
_output_shapes

:@ *
dtype0
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@ p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer2_167840123*
_output_shapes

: *
dtype0
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer3_167840170*
_output_shapes

:*
dtype0
 layer3/kernel/Regularizer/SquareSquare7layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer3/kernel/Regularizer/SumSum$layer3/kernel/Regularizer/Square:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity+prediction/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^layer1/StatefulPartitionedCall0^layer1/kernel/Regularizer/Square/ReadVariableOp^layer2/StatefulPartitionedCall0^layer2/kernel/Regularizer/Square/ReadVariableOp^layer3/StatefulPartitionedCall0^layer3/kernel/Regularizer/Square/ReadVariableOp*^mf_embedding_item/StatefulPartitionedCall?^mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp*^mf_embedding_user/StatefulPartitionedCall?^mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp+^mlp_embedding_item/StatefulPartitionedCall@^mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp+^mlp_embedding_user/StatefulPartitionedCall@^mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp#^prediction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2b
/layer3/kernel/Regularizer/Square/ReadVariableOp/layer3/kernel/Regularizer/Square/ReadVariableOp2V
)mf_embedding_item/StatefulPartitionedCall)mf_embedding_item/StatefulPartitionedCall2
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp2V
)mf_embedding_user/StatefulPartitionedCall)mf_embedding_user/StatefulPartitionedCall2
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp2X
*mlp_embedding_item/StatefulPartitionedCall*mlp_embedding_item/StatefulPartitionedCall2
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp2X
*mlp_embedding_user/StatefulPartitionedCall*mlp_embedding_user/StatefulPartitionedCall2
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp2H
"prediction/StatefulPartitionedCall"prediction/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
ô
Q__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_167840888

inputs-
embedding_lookup_167840876:	/ 
identity¢embedding_lookup¢?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp¿
embedding_lookupResourceGatherembedding_lookup_167840876inputs*
Tindices0*-
_class#
!loc:@embedding_lookup/167840876*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0¥
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/167840876*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_167840876*
_output_shapes
:	/ *
dtype0­
0mlp_embedding_user/embeddings/Regularizer/SquareSquareGmlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/ 
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_user/embeddings/Regularizer/SumSum4mlp_embedding_user/embeddings/Regularizer/Square:y:08mlp_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_user/embeddings/Regularizer/mulMul8mlp_embedding_user/embeddings/Regularizer/mul/x:output:06mlp_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^embedding_lookup@^mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup2
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
ô
Q__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_167840015

inputs-
embedding_lookup_167840003:	/ 
identity¢embedding_lookup¢?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp¿
embedding_lookupResourceGatherembedding_lookup_167840003inputs*
Tindices0*-
_class#
!loc:@embedding_lookup/167840003*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0¥
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/167840003*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_167840003*
_output_shapes
:	/ *
dtype0­
0mlp_embedding_user/embeddings/Regularizer/SquareSquareGmlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/ 
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_user/embeddings/Regularizer/SumSum4mlp_embedding_user/embeddings/Regularizer/Square:y:08mlp_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_user/embeddings/Regularizer/mulMul8mlp_embedding_user/embeddings/Regularizer/mul/x:output:06mlp_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^embedding_lookup@^mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup2
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

°
__inference_loss_fn_5_167841236J
8layer2_kernel_regularizer_square_readvariableop_resource: 
identity¢/layer2/kernel/Regularizer/Square/ReadVariableOp¨
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8layer2_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: *
dtype0
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentity!layer2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^layer2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp
½
v
L__inference_concatenate_1_layer_call_and_return_conditional_losses_167840182

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ú
I__inference_prediction_layer_call_and_return_conditional_losses_167840195

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
Á
)__inference_model_layer_call_fn_167840614
inputs_0
inputs_1
unknown:	ú 
	unknown_0:	/ 
	unknown_1:@ 
	unknown_2: 
	unknown_3:	ú
	unknown_4:	/
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_167840244o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ä
ô
Q__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_167840916

inputs-
embedding_lookup_167840904:	ú 
identity¢embedding_lookup¢?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp¿
embedding_lookupResourceGatherembedding_lookup_167840904inputs*
Tindices0*-
_class#
!loc:@embedding_lookup/167840904*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0¥
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/167840904*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_167840904*
_output_shapes
:	ú *
dtype0­
0mlp_embedding_item/embeddings/Regularizer/SquareSquareGmlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú 
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_item/embeddings/Regularizer/SumSum4mlp_embedding_item/embeddings/Regularizer/Square:y:08mlp_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_item/embeddings/Regularizer/mulMul8mlp_embedding_item/embeddings/Regularizer/mul/x:output:06mlp_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^embedding_lookup@^mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup2
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

°
__inference_loss_fn_6_167841247J
8layer3_kernel_regularizer_square_readvariableop_resource:
identity¢/layer3/kernel/Regularizer/Square/ReadVariableOp¨
/layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8layer3_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype0
 layer3/kernel/Regularizer/SquareSquare7layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer3/kernel/Regularizer/SumSum$layer3/kernel/Regularizer/Square:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentity!layer3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^layer3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/layer3/kernel/Regularizer/Square/ReadVariableOp/layer3/kernel/Regularizer/Square/ReadVariableOp
®
[
/__inference_concatenate_layer_call_fn_167840944
inputs_0
inputs_1
identityÂ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_concatenate_layer_call_and_return_conditional_losses_167840042`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1
Ä
ô
Q__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_167839996

inputs-
embedding_lookup_167839984:	ú 
identity¢embedding_lookup¢?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp¿
embedding_lookupResourceGatherembedding_lookup_167839984inputs*
Tindices0*-
_class#
!loc:@embedding_lookup/167839984*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0¥
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*-
_class#
!loc:@embedding_lookup/167839984*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_167839984*
_output_shapes
:	ú *
dtype0­
0mlp_embedding_item/embeddings/Regularizer/SquareSquareGmlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú 
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_item/embeddings/Regularizer/SumSum4mlp_embedding_item/embeddings/Regularizer/Square:y:08mlp_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_item/embeddings/Regularizer/mulMul8mlp_embedding_item/embeddings/Regularizer/mul/x:output:06mlp_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^embedding_lookup@^mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup2
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
Ñ
__inference_loss_fn_0_167841181[
Hmlp_embedding_user_embeddings_regularizer_square_readvariableop_resource:	/ 
identity¢?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOpÉ
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpHmlp_embedding_user_embeddings_regularizer_square_readvariableop_resource*
_output_shapes
:	/ *
dtype0­
0mlp_embedding_user/embeddings/Regularizer/SquareSquareGmlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/ 
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_user/embeddings/Regularizer/SumSum4mlp_embedding_user/embeddings/Regularizer/Square:y:08mlp_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_user/embeddings/Regularizer/mulMul8mlp_embedding_user/embeddings/Regularizer/mul/x:output:06mlp_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: o
IdentityIdentity1mlp_embedding_user/embeddings/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp@^mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp
­
I
-__inference_flatten_1_layer_call_fn_167841044

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_167840134`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
d
H__inference_flatten_2_layer_call_and_return_conditional_losses_167840025

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
½

6__inference_mlp_embedding_item_layer_call_fn_167840901

inputs
unknown:	ú 
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_167839996s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
Ï
__inference_loss_fn_2_167841203Z
Gmf_embedding_user_embeddings_regularizer_square_readvariableop_resource:	/
identity¢>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOpÇ
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpGmf_embedding_user_embeddings_regularizer_square_readvariableop_resource*
_output_shapes
:	/*
dtype0«
/mf_embedding_user/embeddings/Regularizer/SquareSquareFmf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_user/embeddings/Regularizer/SumSum3mf_embedding_user/embeddings/Regularizer/Square:y:07mf_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_user/embeddings/Regularizer/mulMul7mf_embedding_user/embeddings/Regularizer/mul/x:output:05mf_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentity0mf_embedding_user/embeddings/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp?^mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp
À
d
H__inference_flatten_1_layer_call_and_return_conditional_losses_167840134

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

°
__inference_loss_fn_4_167841225J
8layer1_kernel_regularizer_square_readvariableop_resource:@ 
identity¢/layer1/kernel/Regularizer/Square/ReadVariableOp¨
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8layer1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@ *
dtype0
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@ p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentity!layer1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: x
NoOpNoOp0^layer1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp

¨
E__inference_layer2_layer_call_and_return_conditional_losses_167840122

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/layer2/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^layer2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
»

5__inference_mf_embedding_item_layer_call_fn_167840992

inputs
unknown:	ú
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_mf_embedding_item_layer_call_and_return_conditional_losses_167840082s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
b
F__inference_flatten_layer_call_and_return_conditional_losses_167840142

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
~
¡

D__inference_model_layer_call_and_return_conditional_losses_167840477

inputs
inputs_1/
mlp_embedding_item_167840395:	ú /
mlp_embedding_user_167840398:	/ "
layer1_167840404:@ 
layer1_167840406: .
mf_embedding_item_167840409:	ú.
mf_embedding_user_167840412:	/"
layer2_167840415: 
layer2_167840417:"
layer3_167840423:
layer3_167840425:&
prediction_167840429:"
prediction_167840431:
identity¢layer1/StatefulPartitionedCall¢/layer1/kernel/Regularizer/Square/ReadVariableOp¢layer2/StatefulPartitionedCall¢/layer2/kernel/Regularizer/Square/ReadVariableOp¢layer3/StatefulPartitionedCall¢/layer3/kernel/Regularizer/Square/ReadVariableOp¢)mf_embedding_item/StatefulPartitionedCall¢>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp¢)mf_embedding_user/StatefulPartitionedCall¢>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp¢*mlp_embedding_item/StatefulPartitionedCall¢?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp¢*mlp_embedding_user/StatefulPartitionedCall¢?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp¢"prediction/StatefulPartitionedCall
*mlp_embedding_item/StatefulPartitionedCallStatefulPartitionedCallinputs_1mlp_embedding_item_167840395*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_167839996
*mlp_embedding_user/StatefulPartitionedCallStatefulPartitionedCallinputsmlp_embedding_user_167840398*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_167840015ê
flatten_2/PartitionedCallPartitionedCall3mlp_embedding_user/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_2_layer_call_and_return_conditional_losses_167840025ê
flatten_3/PartitionedCallPartitionedCall3mlp_embedding_item/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_3_layer_call_and_return_conditional_losses_167840033
concatenate/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_concatenate_layer_call_and_return_conditional_losses_167840042
layer1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0layer1_167840404layer1_167840406*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_layer1_layer_call_and_return_conditional_losses_167840061
)mf_embedding_item/StatefulPartitionedCallStatefulPartitionedCallinputs_1mf_embedding_item_167840409*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_mf_embedding_item_layer_call_and_return_conditional_losses_167840082
)mf_embedding_user/StatefulPartitionedCallStatefulPartitionedCallinputsmf_embedding_user_167840412*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_mf_embedding_user_layer_call_and_return_conditional_losses_167840101
layer2/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0layer2_167840415layer2_167840417*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_layer2_layer_call_and_return_conditional_losses_167840122é
flatten_1/PartitionedCallPartitionedCall2mf_embedding_user/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_1_layer_call_and_return_conditional_losses_167840134å
flatten/PartitionedCallPartitionedCall2mf_embedding_item/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_167840142ú
multiply/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_multiply_layer_call_and_return_conditional_losses_167840150
layer3/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0layer3_167840423layer3_167840425*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_layer3_layer_call_and_return_conditional_losses_167840169
concatenate_1/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0'layer3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_concatenate_1_layer_call_and_return_conditional_losses_167840182¡
"prediction/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0prediction_167840429prediction_167840431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_prediction_layer_call_and_return_conditional_losses_167840195
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmlp_embedding_user_167840398*
_output_shapes
:	/ *
dtype0­
0mlp_embedding_user/embeddings/Regularizer/SquareSquareGmlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/ 
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_user/embeddings/Regularizer/SumSum4mlp_embedding_user/embeddings/Regularizer/Square:y:08mlp_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_user/embeddings/Regularizer/mulMul8mlp_embedding_user/embeddings/Regularizer/mul/x:output:06mlp_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmlp_embedding_item_167840395*
_output_shapes
:	ú *
dtype0­
0mlp_embedding_item/embeddings/Regularizer/SquareSquareGmlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú 
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_item/embeddings/Regularizer/SumSum4mlp_embedding_item/embeddings/Regularizer/Square:y:08mlp_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_item/embeddings/Regularizer/mulMul8mlp_embedding_item/embeddings/Regularizer/mul/x:output:06mlp_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmf_embedding_user_167840412*
_output_shapes
:	/*
dtype0«
/mf_embedding_user/embeddings/Regularizer/SquareSquareFmf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_user/embeddings/Regularizer/SumSum3mf_embedding_user/embeddings/Regularizer/Square:y:07mf_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_user/embeddings/Regularizer/mulMul7mf_embedding_user/embeddings/Regularizer/mul/x:output:05mf_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmf_embedding_item_167840409*
_output_shapes
:	ú*
dtype0«
/mf_embedding_item/embeddings/Regularizer/SquareSquareFmf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_item/embeddings/Regularizer/SumSum3mf_embedding_item/embeddings/Regularizer/Square:y:07mf_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_item/embeddings/Regularizer/mulMul7mf_embedding_item/embeddings/Regularizer/mul/x:output:05mf_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer1_167840404*
_output_shapes

:@ *
dtype0
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@ p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer2_167840415*
_output_shapes

: *
dtype0
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOplayer3_167840423*
_output_shapes

:*
dtype0
 layer3/kernel/Regularizer/SquareSquare7layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer3/kernel/Regularizer/SumSum$layer3/kernel/Regularizer/Square:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity+prediction/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^layer1/StatefulPartitionedCall0^layer1/kernel/Regularizer/Square/ReadVariableOp^layer2/StatefulPartitionedCall0^layer2/kernel/Regularizer/Square/ReadVariableOp^layer3/StatefulPartitionedCall0^layer3/kernel/Regularizer/Square/ReadVariableOp*^mf_embedding_item/StatefulPartitionedCall?^mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp*^mf_embedding_user/StatefulPartitionedCall?^mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp+^mlp_embedding_item/StatefulPartitionedCall@^mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp+^mlp_embedding_user/StatefulPartitionedCall@^mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp#^prediction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp2@
layer3/StatefulPartitionedCalllayer3/StatefulPartitionedCall2b
/layer3/kernel/Regularizer/Square/ReadVariableOp/layer3/kernel/Regularizer/Square/ReadVariableOp2V
)mf_embedding_item/StatefulPartitionedCall)mf_embedding_item/StatefulPartitionedCall2
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp2V
)mf_embedding_user/StatefulPartitionedCall)mf_embedding_user/StatefulPartitionedCall2
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp2X
*mlp_embedding_item/StatefulPartitionedCall*mlp_embedding_item/StatefulPartitionedCall2
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp2X
*mlp_embedding_user/StatefulPartitionedCall*mlp_embedding_user/StatefulPartitionedCall2
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp2H
"prediction/StatefulPartitionedCall"prediction/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¨
E__inference_layer2_layer_call_and_return_conditional_losses_167841093

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/layer2/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^layer2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ñ
½
'__inference_signature_wrapper_167840584

args_0
args_0_1
unknown:	ú 
	unknown_0:	/ 
	unknown_1:@ 
	unknown_2: 
	unknown_3:	ú
	unknown_4:	/
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__wrapped_model_167839972o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
args_0_1
Â
s
G__inference_multiply_layer_call_and_return_conditional_losses_167841105
inputs_0
inputs_1
identityP
mulMulinputs_0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
©
Ï
__inference_loss_fn_3_167841214Z
Gmf_embedding_item_embeddings_regularizer_square_readvariableop_resource:	ú
identity¢>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOpÇ
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpGmf_embedding_item_embeddings_regularizer_square_readvariableop_resource*
_output_shapes
:	ú*
dtype0«
/mf_embedding_item/embeddings/Regularizer/SquareSquareFmf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_item/embeddings/Regularizer/SumSum3mf_embedding_item/embeddings/Regularizer/Square:y:07mf_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_item/embeddings/Regularizer/mulMul7mf_embedding_item/embeddings/Regularizer/mul/x:output:05mf_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: n
IdentityIdentity0mf_embedding_item/embeddings/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp?^mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp
ù
Á
)__inference_model_layer_call_fn_167840644
inputs_0
inputs_1
unknown:	ú 
	unknown_0:	/ 
	unknown_1:@ 
	unknown_2: 
	unknown_3:	ú
	unknown_4:	/
	unknown_5: 
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_167840477o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ã
v
J__inference_concatenate_layer_call_and_return_conditional_losses_167840951
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1
»

5__inference_mf_embedding_user_layer_call_fn_167840964

inputs
unknown:	/
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_mf_embedding_user_layer_call_and_return_conditional_losses_167840101s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¨
E__inference_layer1_layer_call_and_return_conditional_losses_167841039

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/layer1/kernel/Regularizer/Square/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@ p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ©
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ä

*__inference_layer1_layer_call_fn_167841022

inputs
unknown:@ 
	unknown_0: 
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_layer1_layer_call_and_return_conditional_losses_167840061o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¼
Ñ
__inference_loss_fn_1_167841192[
Hmlp_embedding_item_embeddings_regularizer_square_readvariableop_resource:	ú 
identity¢?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOpÉ
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpHmlp_embedding_item_embeddings_regularizer_square_readvariableop_resource*
_output_shapes
:	ú *
dtype0­
0mlp_embedding_item/embeddings/Regularizer/SquareSquareGmlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú 
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_item/embeddings/Regularizer/SumSum4mlp_embedding_item/embeddings/Regularizer/Square:y:08mlp_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_item/embeddings/Regularizer/mulMul8mlp_embedding_item/embeddings/Regularizer/mul/x:output:06mlp_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: o
IdentityIdentity1mlp_embedding_item/embeddings/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp@^mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp
µ
ó
D__inference_model_layer_call_and_return_conditional_losses_167840860
inputs_0
inputs_1@
-mlp_embedding_item_embedding_lookup_167840756:	ú @
-mlp_embedding_user_embedding_lookup_167840761:	/ 7
%layer1_matmul_readvariableop_resource:@ 4
&layer1_biasadd_readvariableop_resource: ?
,mf_embedding_item_embedding_lookup_167840779:	ú?
,mf_embedding_user_embedding_lookup_167840784:	/7
%layer2_matmul_readvariableop_resource: 4
&layer2_biasadd_readvariableop_resource:7
%layer3_matmul_readvariableop_resource:4
&layer3_biasadd_readvariableop_resource:;
)prediction_matmul_readvariableop_resource:8
*prediction_biasadd_readvariableop_resource:
identity¢layer1/BiasAdd/ReadVariableOp¢layer1/MatMul/ReadVariableOp¢/layer1/kernel/Regularizer/Square/ReadVariableOp¢layer2/BiasAdd/ReadVariableOp¢layer2/MatMul/ReadVariableOp¢/layer2/kernel/Regularizer/Square/ReadVariableOp¢layer3/BiasAdd/ReadVariableOp¢layer3/MatMul/ReadVariableOp¢/layer3/kernel/Regularizer/Square/ReadVariableOp¢"mf_embedding_item/embedding_lookup¢>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp¢"mf_embedding_user/embedding_lookup¢>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp¢#mlp_embedding_item/embedding_lookup¢?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp¢#mlp_embedding_user/embedding_lookup¢?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp¢!prediction/BiasAdd/ReadVariableOp¢ prediction/MatMul/ReadVariableOpú
#mlp_embedding_item/embedding_lookupResourceGather-mlp_embedding_item_embedding_lookup_167840756inputs_1*
Tindices0*@
_class6
42loc:@mlp_embedding_item/embedding_lookup/167840756*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0Þ
,mlp_embedding_item/embedding_lookup/IdentityIdentity,mlp_embedding_item/embedding_lookup:output:0*
T0*@
_class6
42loc:@mlp_embedding_item/embedding_lookup/167840756*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §
.mlp_embedding_item/embedding_lookup/Identity_1Identity5mlp_embedding_item/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ú
#mlp_embedding_user/embedding_lookupResourceGather-mlp_embedding_user_embedding_lookup_167840761inputs_0*
Tindices0*@
_class6
42loc:@mlp_embedding_user/embedding_lookup/167840761*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0Þ
,mlp_embedding_user/embedding_lookup/IdentityIdentity,mlp_embedding_user/embedding_lookup:output:0*
T0*@
_class6
42loc:@mlp_embedding_user/embedding_lookup/167840761*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §
.mlp_embedding_user/embedding_lookup/Identity_1Identity5mlp_embedding_user/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¡
flatten_2/ReshapeReshape7mlp_embedding_user/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    ¡
flatten_3/ReshapeReshape7mlp_embedding_item/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :³
concatenate/concatConcatV2flatten_2/Reshape:output:0flatten_3/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
layer1/MatMulMatMulconcatenate/concat:output:0$layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ^
layer1/ReluRelulayer1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ÷
"mf_embedding_item/embedding_lookupResourceGather,mf_embedding_item_embedding_lookup_167840779inputs_1*
Tindices0*?
_class5
31loc:@mf_embedding_item/embedding_lookup/167840779*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Û
+mf_embedding_item/embedding_lookup/IdentityIdentity+mf_embedding_item/embedding_lookup:output:0*
T0*?
_class5
31loc:@mf_embedding_item/embedding_lookup/167840779*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-mf_embedding_item/embedding_lookup/Identity_1Identity4mf_embedding_item/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ÷
"mf_embedding_user/embedding_lookupResourceGather,mf_embedding_user_embedding_lookup_167840784inputs_0*
Tindices0*?
_class5
31loc:@mf_embedding_user/embedding_lookup/167840784*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Û
+mf_embedding_user/embedding_lookup/IdentityIdentity+mf_embedding_user/embedding_lookup:output:0*
T0*?
_class5
31loc:@mf_embedding_user/embedding_lookup/167840784*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
-mf_embedding_user/embedding_lookup/Identity_1Identity4mf_embedding_user/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
layer2/MatMulMatMullayer1/Relu:activations:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
layer2/ReluRelulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    
flatten_1/ReshapeReshape6mf_embedding_user/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten/ReshapeReshape6mf_embedding_item/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
multiply/mulMulflatten_1/Reshape:output:0flatten/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
layer3/MatMul/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
layer3/MatMulMatMullayer2/Relu:activations:0$layer3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
layer3/BiasAdd/ReadVariableOpReadVariableOp&layer3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
layer3/BiasAddBiasAddlayer3/MatMul:product:0%layer3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
layer3/ReluRelulayer3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¬
concatenate_1/concatConcatV2multiply/mul:z:0layer3/Relu:activations:0"concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 prediction/MatMul/ReadVariableOpReadVariableOp)prediction_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
prediction/MatMulMatMulconcatenate_1/concat:output:0(prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!prediction/BiasAdd/ReadVariableOpReadVariableOp*prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
prediction/BiasAddBiasAddprediction/MatMul:product:0)prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
prediction/SigmoidSigmoidprediction/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp-mlp_embedding_user_embedding_lookup_167840761*
_output_shapes
:	/ *
dtype0­
0mlp_embedding_user/embeddings/Regularizer/SquareSquareGmlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/ 
/mlp_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_user/embeddings/Regularizer/SumSum4mlp_embedding_user/embeddings/Regularizer/Square:y:08mlp_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_user/embeddings/Regularizer/mulMul8mlp_embedding_user/embeddings/Regularizer/mul/x:output:06mlp_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: ®
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp-mlp_embedding_item_embedding_lookup_167840756*
_output_shapes
:	ú *
dtype0­
0mlp_embedding_item/embeddings/Regularizer/SquareSquareGmlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú 
/mlp_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Å
-mlp_embedding_item/embeddings/Regularizer/SumSum4mlp_embedding_item/embeddings/Regularizer/Square:y:08mlp_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: t
/mlp_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ç
-mlp_embedding_item/embeddings/Regularizer/mulMul8mlp_embedding_item/embeddings/Regularizer/mul/x:output:06mlp_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¬
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp,mf_embedding_user_embedding_lookup_167840784*
_output_shapes
:	/*
dtype0«
/mf_embedding_user/embeddings/Regularizer/SquareSquareFmf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	/
.mf_embedding_user/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_user/embeddings/Regularizer/SumSum3mf_embedding_user/embeddings/Regularizer/Square:y:07mf_embedding_user/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_user/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_user/embeddings/Regularizer/mulMul7mf_embedding_user/embeddings/Regularizer/mul/x:output:05mf_embedding_user/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: ¬
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp,mf_embedding_item_embedding_lookup_167840779*
_output_shapes
:	ú*
dtype0«
/mf_embedding_item/embeddings/Regularizer/SquareSquareFmf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ú
.mf_embedding_item/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Â
,mf_embedding_item/embeddings/Regularizer/SumSum3mf_embedding_item/embeddings/Regularizer/Square:y:07mf_embedding_item/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: s
.mf_embedding_item/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ä
,mf_embedding_item/embeddings/Regularizer/mulMul7mf_embedding_item/embeddings/Regularizer/mul/x:output:05mf_embedding_item/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/layer1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
 layer1/kernel/Regularizer/SquareSquare7layer1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@ p
layer1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer1/kernel/Regularizer/SumSum$layer1/kernel/Regularizer/Square:y:0(layer1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer1/kernel/Regularizer/mulMul(layer1/kernel/Regularizer/mul/x:output:0&layer1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/layer2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
 layer2/kernel/Regularizer/SquareSquare7layer2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: p
layer2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer2/kernel/Regularizer/SumSum$layer2/kernel/Regularizer/Square:y:0(layer2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer2/kernel/Regularizer/mulMul(layer2/kernel/Regularizer/mul/x:output:0&layer2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
/layer3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%layer3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
 layer3/kernel/Regularizer/SquareSquare7layer3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:p
layer3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
layer3/kernel/Regularizer/SumSum$layer3/kernel/Regularizer/Square:y:0(layer3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: d
layer3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
layer3/kernel/Regularizer/mulMul(layer3/kernel/Regularizer/mul/x:output:0&layer3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentityprediction/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
NoOpNoOp^layer1/BiasAdd/ReadVariableOp^layer1/MatMul/ReadVariableOp0^layer1/kernel/Regularizer/Square/ReadVariableOp^layer2/BiasAdd/ReadVariableOp^layer2/MatMul/ReadVariableOp0^layer2/kernel/Regularizer/Square/ReadVariableOp^layer3/BiasAdd/ReadVariableOp^layer3/MatMul/ReadVariableOp0^layer3/kernel/Regularizer/Square/ReadVariableOp#^mf_embedding_item/embedding_lookup?^mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp#^mf_embedding_user/embedding_lookup?^mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp$^mlp_embedding_item/embedding_lookup@^mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp$^mlp_embedding_user/embedding_lookup@^mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp"^prediction/BiasAdd/ReadVariableOp!^prediction/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
layer1/BiasAdd/ReadVariableOplayer1/BiasAdd/ReadVariableOp2<
layer1/MatMul/ReadVariableOplayer1/MatMul/ReadVariableOp2b
/layer1/kernel/Regularizer/Square/ReadVariableOp/layer1/kernel/Regularizer/Square/ReadVariableOp2>
layer2/BiasAdd/ReadVariableOplayer2/BiasAdd/ReadVariableOp2<
layer2/MatMul/ReadVariableOplayer2/MatMul/ReadVariableOp2b
/layer2/kernel/Regularizer/Square/ReadVariableOp/layer2/kernel/Regularizer/Square/ReadVariableOp2>
layer3/BiasAdd/ReadVariableOplayer3/BiasAdd/ReadVariableOp2<
layer3/MatMul/ReadVariableOplayer3/MatMul/ReadVariableOp2b
/layer3/kernel/Regularizer/Square/ReadVariableOp/layer3/kernel/Regularizer/Square/ReadVariableOp2H
"mf_embedding_item/embedding_lookup"mf_embedding_item/embedding_lookup2
>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_item/embeddings/Regularizer/Square/ReadVariableOp2H
"mf_embedding_user/embedding_lookup"mf_embedding_user/embedding_lookup2
>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp>mf_embedding_user/embeddings/Regularizer/Square/ReadVariableOp2J
#mlp_embedding_item/embedding_lookup#mlp_embedding_item/embedding_lookup2
?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_item/embeddings/Regularizer/Square/ReadVariableOp2J
#mlp_embedding_user/embedding_lookup#mlp_embedding_user/embedding_lookup2
?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp?mlp_embedding_user/embeddings/Regularizer/Square/ReadVariableOp2F
!prediction/BiasAdd/ReadVariableOp!prediction/BiasAdd/ReadVariableOp2D
 prediction/MatMul/ReadVariableOp prediction/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
À
d
H__inference_flatten_3_layer_call_and_return_conditional_losses_167840938

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
©
G
+__inference_flatten_layer_call_fn_167841055

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_layer_call_and_return_conditional_losses_167840142`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
t
J__inference_concatenate_layer_call_and_return_conditional_losses_167840042

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
À
d
H__inference_flatten_2_layer_call_and_return_conditional_losses_167840927

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ    \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¯
½
%__inference__traced_restore_167841539
file_prefixA
.assignvariableop_mlp_embedding_user_embeddings:	/ C
0assignvariableop_1_mlp_embedding_item_embeddings:	ú B
/assignvariableop_2_mf_embedding_user_embeddings:	/B
/assignvariableop_3_mf_embedding_item_embeddings:	ú2
 assignvariableop_4_layer1_kernel:@ ,
assignvariableop_5_layer1_bias: 2
 assignvariableop_6_layer2_kernel: ,
assignvariableop_7_layer2_bias:2
 assignvariableop_8_layer3_kernel:,
assignvariableop_9_layer3_bias:7
%assignvariableop_10_prediction_kernel:1
#assignvariableop_11_prediction_bias:'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: K
8assignvariableop_19_adam_mlp_embedding_user_embeddings_m:	/ K
8assignvariableop_20_adam_mlp_embedding_item_embeddings_m:	ú J
7assignvariableop_21_adam_mf_embedding_user_embeddings_m:	/J
7assignvariableop_22_adam_mf_embedding_item_embeddings_m:	ú:
(assignvariableop_23_adam_layer1_kernel_m:@ 4
&assignvariableop_24_adam_layer1_bias_m: :
(assignvariableop_25_adam_layer2_kernel_m: 4
&assignvariableop_26_adam_layer2_bias_m::
(assignvariableop_27_adam_layer3_kernel_m:4
&assignvariableop_28_adam_layer3_bias_m:>
,assignvariableop_29_adam_prediction_kernel_m:8
*assignvariableop_30_adam_prediction_bias_m:K
8assignvariableop_31_adam_mlp_embedding_user_embeddings_v:	/ K
8assignvariableop_32_adam_mlp_embedding_item_embeddings_v:	ú J
7assignvariableop_33_adam_mf_embedding_user_embeddings_v:	/J
7assignvariableop_34_adam_mf_embedding_item_embeddings_v:	ú:
(assignvariableop_35_adam_layer1_kernel_v:@ 4
&assignvariableop_36_adam_layer1_bias_v: :
(assignvariableop_37_adam_layer2_kernel_v: 4
&assignvariableop_38_adam_layer2_bias_v::
(assignvariableop_39_adam_layer3_kernel_v:4
&assignvariableop_40_adam_layer3_bias_v:>
,assignvariableop_41_adam_prediction_kernel_v:8
*assignvariableop_42_adam_prediction_bias_v:
identity_44¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ö
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*
valueB,B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÈ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ý
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Æ
_output_shapes³
°::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp.assignvariableop_mlp_embedding_user_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp0assignvariableop_1_mlp_embedding_item_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp/assignvariableop_2_mf_embedding_user_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp/assignvariableop_3_mf_embedding_item_embeddingsIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_layer1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp assignvariableop_6_layer2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_layer2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp assignvariableop_8_layer3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_layer3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp%assignvariableop_10_prediction_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp#assignvariableop_11_prediction_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_19AssignVariableOp8assignvariableop_19_adam_mlp_embedding_user_embeddings_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_20AssignVariableOp8assignvariableop_20_adam_mlp_embedding_item_embeddings_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adam_mf_embedding_user_embeddings_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_22AssignVariableOp7assignvariableop_22_adam_mf_embedding_item_embeddings_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_layer1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_layer1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_layer2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_layer2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_layer3_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_layer3_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp,assignvariableop_29_adam_prediction_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_prediction_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_31AssignVariableOp8assignvariableop_31_adam_mlp_embedding_user_embeddings_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_32AssignVariableOp8assignvariableop_32_adam_mlp_embedding_item_embeddings_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_mf_embedding_user_embeddings_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_mf_embedding_item_embeddings_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_layer1_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_layer1_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_layer2_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp&assignvariableop_38_adam_layer2_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_layer3_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_layer3_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_prediction_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_prediction_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_44IdentityIdentity_43:output:0^NoOp_1*
T0*
_output_shapes
: î
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_44Identity_44:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
º
q
G__inference_multiply_layer_call_and_return_conditional_losses_167840150

inputs
inputs_1
identityN
mulMulinputsinputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
IdentityIdentitymul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
I
-__inference_flatten_2_layer_call_fn_167840921

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_flatten_2_layer_call_and_return_conditional_losses_167840025`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ê
serving_defaultÖ
9
args_0/
serving_default_args_0:0ÿÿÿÿÿÿÿÿÿ
=
args_0_11
serving_default_args_0_1:0ÿÿÿÿÿÿÿÿÿ>

prediction0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ö

layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
Ò__call__
+Ó&call_and_return_all_conditional_losses
Ô_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
·

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_tf_keras_layer
·

embeddings
	variables
trainable_variables
 regularization_losses
!	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"
_tf_keras_layer
§
"	variables
#trainable_variables
$regularization_losses
%	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses"
_tf_keras_layer
§
&	variables
'trainable_variables
(regularization_losses
)	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses"
_tf_keras_layer
§
*	variables
+trainable_variables
,regularization_losses
-	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"
_tf_keras_layer
·
.
embeddings
/	variables
0trainable_variables
1regularization_losses
2	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"
_tf_keras_layer
·
3
embeddings
4	variables
5trainable_variables
6regularization_losses
7	keras_api
á__call__
+â&call_and_return_all_conditional_losses"
_tf_keras_layer
½

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses"
_tf_keras_layer
§
>	variables
?trainable_variables
@regularization_losses
A	keras_api
å__call__
+æ&call_and_return_all_conditional_losses"
_tf_keras_layer
§
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
ç__call__
+è&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
é__call__
+ê&call_and_return_all_conditional_losses"
_tf_keras_layer
§
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Pkernel
Qbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
í__call__
+î&call_and_return_all_conditional_losses"
_tf_keras_layer
§
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
ï__call__
+ð&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Zkernel
[bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
`iter

abeta_1

bbeta_2
	cdecay
dlearning_ratemºm».m¼3m½8m¾9m¿FmÀGmÁPmÂQmÃZmÄ[mÅvÆvÇ.vÈ3vÉ8vÊ9vËFvÌGvÍPvÎQvÏZvÐ[vÑ"
	optimizer
v
0
1
.2
33
84
95
F6
G7
P8
Q9
Z10
[11"
trackable_list_wrapper
v
0
1
.2
33
84
95
F6
G7
P8
Q9
Z10
[11"
trackable_list_wrapper
X
ó0
ô1
õ2
ö3
÷4
ø5
ù6"
trackable_list_wrapper
Î
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
Ò__call__
Ô_default_save_signature
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
-
úserving_default"
signature_map
0:.	/ 2mlp_embedding_user/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
(
ó0"
trackable_list_wrapper
°
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
0:.	ú 2mlp_embedding_item/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
(
ô0"
trackable_list_wrapper
°
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
 regularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
tnon_trainable_variables

ulayers
vmetrics
wlayer_regularization_losses
xlayer_metrics
"	variables
#trainable_variables
$regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
&	variables
'trainable_variables
(regularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
³
~non_trainable_variables

layers
metrics
 layer_regularization_losses
layer_metrics
*	variables
+trainable_variables
,regularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
/:-	/2mf_embedding_user/embeddings
'
.0"
trackable_list_wrapper
'
.0"
trackable_list_wrapper
(
õ0"
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
/	variables
0trainable_variables
1regularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
/:-	ú2mf_embedding_item/embeddings
'
30"
trackable_list_wrapper
'
30"
trackable_list_wrapper
(
ö0"
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
4	variables
5trainable_variables
6regularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
:@ 2layer1/kernel
: 2layer1/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
(
÷0"
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
:	variables
;trainable_variables
<regularization_losses
ã__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
>	variables
?trainable_variables
@regularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
: 2layer2/kernel
:2layer2/bias
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
(
ø0"
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
é__call__
+ê&call_and_return_all_conditional_losses
'ê"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
:2layer3/kernel
:2layer3/bias
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
(
ù0"
trackable_list_wrapper
µ
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
«non_trainable_variables
¬layers
­metrics
 ®layer_regularization_losses
¯layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
ï__call__
+ð&call_and_return_all_conditional_losses
'ð"call_and_return_conditional_losses"
_generic_user_object
#:!2prediction/kernel
:2prediction/bias
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
°non_trainable_variables
±layers
²metrics
 ³layer_regularization_losses
´layer_metrics
\	variables
]trainable_variables
^regularization_losses
ñ__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
(
µ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ó0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ô0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
õ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ö0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
÷0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ø0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ù0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

¶total

·count
¸	variables
¹	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
¶0
·1"
trackable_list_wrapper
.
¸	variables"
_generic_user_object
5:3	/ 2$Adam/mlp_embedding_user/embeddings/m
5:3	ú 2$Adam/mlp_embedding_item/embeddings/m
4:2	/2#Adam/mf_embedding_user/embeddings/m
4:2	ú2#Adam/mf_embedding_item/embeddings/m
$:"@ 2Adam/layer1/kernel/m
: 2Adam/layer1/bias/m
$:" 2Adam/layer2/kernel/m
:2Adam/layer2/bias/m
$:"2Adam/layer3/kernel/m
:2Adam/layer3/bias/m
(:&2Adam/prediction/kernel/m
": 2Adam/prediction/bias/m
5:3	/ 2$Adam/mlp_embedding_user/embeddings/v
5:3	ú 2$Adam/mlp_embedding_item/embeddings/v
4:2	/2#Adam/mf_embedding_user/embeddings/v
4:2	ú2#Adam/mf_embedding_item/embeddings/v
$:"@ 2Adam/layer1/kernel/v
: 2Adam/layer1/bias/v
$:" 2Adam/layer2/kernel/v
:2Adam/layer2/bias/v
$:"2Adam/layer3/kernel/v
:2Adam/layer3/bias/v
(:&2Adam/prediction/kernel/v
": 2Adam/prediction/bias/v
2
)__inference_model_layer_call_fn_167840614
)__inference_model_layer_call_fn_167840644À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
D__inference_model_layer_call_and_return_conditional_losses_167840752
D__inference_model_layer_call_and_return_conditional_losses_167840860À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ØBÕ
$__inference__wrapped_model_167839972args_0args_0_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
à2Ý
6__inference_mlp_embedding_user_layer_call_fn_167840873¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
û2ø
Q__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_167840888¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
à2Ý
6__inference_mlp_embedding_item_layer_call_fn_167840901¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
û2ø
Q__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_167840916¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_flatten_2_layer_call_fn_167840921¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_flatten_2_layer_call_and_return_conditional_losses_167840927¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_flatten_3_layer_call_fn_167840932¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_flatten_3_layer_call_and_return_conditional_losses_167840938¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ù2Ö
/__inference_concatenate_layer_call_fn_167840944¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_concatenate_layer_call_and_return_conditional_losses_167840951¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_mf_embedding_user_layer_call_fn_167840964¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_mf_embedding_user_layer_call_and_return_conditional_losses_167840979¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_mf_embedding_item_layer_call_fn_167840992¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_mf_embedding_item_layer_call_and_return_conditional_losses_167841007¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_layer1_layer_call_fn_167841022¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_layer1_layer_call_and_return_conditional_losses_167841039¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_flatten_1_layer_call_fn_167841044¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_flatten_1_layer_call_and_return_conditional_losses_167841050¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_flatten_layer_call_fn_167841055¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_flatten_layer_call_and_return_conditional_losses_167841061¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_layer2_layer_call_fn_167841076¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_layer2_layer_call_and_return_conditional_losses_167841093¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_multiply_layer_call_fn_167841099¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_multiply_layer_call_and_return_conditional_losses_167841105¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_layer3_layer_call_fn_167841120¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_layer3_layer_call_and_return_conditional_losses_167841137¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_concatenate_1_layer_call_fn_167841143¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_concatenate_1_layer_call_and_return_conditional_losses_167841150¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_prediction_layer_call_fn_167841159¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_prediction_layer_call_and_return_conditional_losses_167841170¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¶2³
__inference_loss_fn_0_167841181
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
¶2³
__inference_loss_fn_1_167841192
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
¶2³
__inference_loss_fn_2_167841203
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
¶2³
__inference_loss_fn_3_167841214
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
¶2³
__inference_loss_fn_4_167841225
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
¶2³
__inference_loss_fn_5_167841236
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
¶2³
__inference_loss_fn_6_167841247
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
ÕBÒ
'__inference_signature_wrapper_167840584args_0args_0_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 Ì
$__inference__wrapped_model_167839972£893.FGPQZ[Z¢W
P¢M
KH
"
args_0/0ÿÿÿÿÿÿÿÿÿ
"
args_0/1ÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

prediction$!

predictionÿÿÿÿÿÿÿÿÿÔ
L__inference_concatenate_1_layer_call_and_return_conditional_losses_167841150Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 «
1__inference_concatenate_1_layer_call_fn_167841143vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÒ
J__inference_concatenate_layer_call_and_return_conditional_losses_167840951Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ 
"
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ©
/__inference_concatenate_layer_call_fn_167840944vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ 
"
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ@¨
H__inference_flatten_1_layer_call_and_return_conditional_losses_167841050\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_flatten_1_layer_call_fn_167841044O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
H__inference_flatten_2_layer_call_and_return_conditional_losses_167840927\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
-__inference_flatten_2_layer_call_fn_167840921O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¨
H__inference_flatten_3_layer_call_and_return_conditional_losses_167840938\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
-__inference_flatten_3_layer_call_fn_167840932O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¦
F__inference_flatten_layer_call_and_return_conditional_losses_167841061\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_flatten_layer_call_fn_167841055O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_layer1_layer_call_and_return_conditional_losses_167841039\89/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 }
*__inference_layer1_layer_call_fn_167841022O89/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ ¥
E__inference_layer2_layer_call_and_return_conditional_losses_167841093\FG/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_layer2_layer_call_fn_167841076OFG/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_layer3_layer_call_and_return_conditional_losses_167841137\PQ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_layer3_layer_call_fn_167841120OPQ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ>
__inference_loss_fn_0_167841181¢

¢ 
ª " >
__inference_loss_fn_1_167841192¢

¢ 
ª " >
__inference_loss_fn_2_167841203.¢

¢ 
ª " >
__inference_loss_fn_3_1678412143¢

¢ 
ª " >
__inference_loss_fn_4_1678412258¢

¢ 
ª " >
__inference_loss_fn_5_167841236F¢

¢ 
ª " >
__inference_loss_fn_6_167841247P¢

¢ 
ª " ³
P__inference_mf_embedding_item_layer_call_and_return_conditional_losses_167841007_3/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
5__inference_mf_embedding_item_layer_call_fn_167840992R3/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ³
P__inference_mf_embedding_user_layer_call_and_return_conditional_losses_167840979_./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
5__inference_mf_embedding_user_layer_call_fn_167840964R./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ´
Q__inference_mlp_embedding_item_layer_call_and_return_conditional_losses_167840916_/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 
6__inference_mlp_embedding_item_layer_call_fn_167840901R/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ ´
Q__inference_mlp_embedding_user_layer_call_and_return_conditional_losses_167840888_/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ 
 
6__inference_mlp_embedding_user_layer_call_fn_167840873R/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ â
D__inference_model_layer_call_and_return_conditional_losses_167840752893.FGPQZ[b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 â
D__inference_model_layer_call_and_return_conditional_losses_167840860893.FGPQZ[b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
)__inference_model_layer_call_fn_167840614893.FGPQZ[b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿº
)__inference_model_layer_call_fn_167840644893.FGPQZ[b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÏ
G__inference_multiply_layer_call_and_return_conditional_losses_167841105Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¦
,__inference_multiply_layer_call_fn_167841099vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
I__inference_prediction_layer_call_and_return_conditional_losses_167841170\Z[/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_prediction_layer_call_fn_167841159OZ[/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÞ
'__inference_signature_wrapper_167840584²893.FGPQZ[i¢f
¢ 
_ª\
*
args_0 
args_0ÿÿÿÿÿÿÿÿÿ
.
args_0_1"
args_0_1ÿÿÿÿÿÿÿÿÿ"7ª4
2

prediction$!

predictionÿÿÿÿÿÿÿÿÿ