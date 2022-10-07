
�yroot"_tf_keras_network*�x{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "item_input"}, "name": "item_input", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "mlp_embedding_user", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 6040, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mlp_embedding_user", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "mlp_embedding_item", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3706, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mlp_embedding_item", "inbound_nodes": [[["item_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["mlp_embedding_user", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["mlp_embedding_item", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["flatten_2", 0, 0, {}], ["flatten_3", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "mf_embedding_user", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 6040, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mf_embedding_user", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "mf_embedding_item", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3706, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mf_embedding_item", "inbound_nodes": [[["item_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["mf_embedding_user", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["mf_embedding_item", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2", "inbound_nodes": [[["layer1", 0, 0, {}]]]}, {"class_name": "Multiply", "config": {"name": "multiply", "trainable": true, "dtype": "float32"}, "name": "multiply", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3", "inbound_nodes": [[["layer2", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["multiply", 0, 0, {}], ["layer3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "prediction", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "LecunUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "prediction", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["item_input", 0, 0]], "output_layers": [["prediction", 0, 0]]}, "shared_object_id": 36, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "int32", null]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "int32", "item_input"]}]], {}]}, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "int32", null]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "int32", "item_input"]}], "keras_version": "2.7.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "item_input"}, "name": "item_input", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "Embedding", "config": {"name": "mlp_embedding_user", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 6040, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 3}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mlp_embedding_user", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Embedding", "config": {"name": "mlp_embedding_item", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3706, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 5}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 6}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mlp_embedding_item", "inbound_nodes": [[["item_input", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["mlp_embedding_user", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["mlp_embedding_item", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["flatten_2", 0, 0, {}], ["flatten_3", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Embedding", "config": {"name": "mf_embedding_user", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 6040, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 11}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 12}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mf_embedding_user", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Embedding", "config": {"name": "mf_embedding_item", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3706, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 14}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 15}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "mf_embedding_item", "inbound_nodes": [[["item_input", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "Dense", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 19}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer1", "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["mf_embedding_user", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["mf_embedding_item", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Dense", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 25}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer2", "inbound_nodes": [[["layer1", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "Multiply", "config": {"name": "multiply", "trainable": true, "dtype": "float32"}, "name": "multiply", "inbound_nodes": [[["flatten_1", 0, 0, {}], ["flatten", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "Dense", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 30}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "layer3", "inbound_nodes": [[["layer2", 0, 0, {}]]], "shared_object_id": 31}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["multiply", 0, 0, {}], ["layer3", 0, 0, {}]]], "shared_object_id": 32}, {"class_name": "Dense", "config": {"name": "prediction", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "LecunUniform", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "prediction", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]], "shared_object_id": 35}], "input_layers": [["input_1", 0, 0], ["item_input", 0, 0]], "output_layers": [["prediction", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_1", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "input_1"}}2
�root.layer-1"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "item_input", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "item_input"}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "mlp_embedding_user", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "mlp_embedding_user", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 6040, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 3}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 4, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "mlp_embedding_item", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "mlp_embedding_item", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3706, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 5}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 6}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "inbound_nodes": [[["item_input", 0, 0, {}]]], "shared_object_id": 7, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}2
�root.layer-4"_tf_keras_layer*�{"name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["mlp_embedding_user", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 39}}2
�root.layer-5"_tf_keras_layer*�{"name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["mlp_embedding_item", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 40}}2
�root.layer-6"_tf_keras_layer*�{"name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["flatten_2", 0, 0, {}], ["flatten_3", 0, 0, {}]]], "shared_object_id": 10, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 32]}]}2
�root.layer_with_weights-2"_tf_keras_layer*�{"name": "mf_embedding_user", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "mf_embedding_user", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 6040, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 11}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 12}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 13, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}2
�	root.layer_with_weights-3"_tf_keras_layer*�{"name": "mf_embedding_item", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "mf_embedding_item", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3706, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}, "shared_object_id": 14}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 15}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "inbound_nodes": [[["item_input", 0, 0, {}]]], "shared_object_id": 16, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}2
�
root.layer_with_weights-4"_tf_keras_layer*�{"name": "layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 19}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}2
�
�
�
�
�root.layer_with_weights-6"_tf_keras_layer*�{"name": "layer3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "layer3", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 30}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["layer2", 0, 0, {}]]], "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}2
�
�root.layer_with_weights-7"_tf_keras_layer*�{"name": "prediction", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "prediction", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "LecunUniform", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_1", 0, 0, {}]]], "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}2
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 47}2