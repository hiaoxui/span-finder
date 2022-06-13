local dataset_path = std.extVar("DATA_PATH");
local ontology_path = std.extVar("ONTOLOGY_PATH");

local debug = false;

# reader
local pretrained_model = std.extVar("ENCODER");
local smoothing_factor = 0.1;

# model
local label_dim = 64;
local dropout = 0.2;
local bio_dim = 512;
local bio_layers = 2;
local span_typing_dims = [256, 256];
local typing_loss_factor = 8.0;

# loader
local exemplar_ratio = 0.05;
local max_training_tokens = 512;
local max_inference_tokens = 1024;

# training
local layer_fix = 0;
local grad_acc = 1;
local cuda_devices = std.parseJson(std.extVar("CUDA"));
local patience = null;

{
    dataset_reader: {
        type: "semantic_role_labeling",
        debug: debug,
        pretrained_model: pretrained_model,
        ignore_label: false,
        [ if debug then "max_instances" ]: 128,
        event_smoothing_factor: smoothing_factor,
        arg_smoothing_factor: smoothing_factor,
    },
    train_data_path: dataset_path + "/train.jsonl",
    validation_data_path: dataset_path + "/dev.jsonl",
    test_data_path: dataset_path + "/test.jsonl",

    datasets_for_vocab_creation: ["train"],

    data_loader: {
        batch_sampler: {
            type: "mix_sampler",
            max_tokens: max_training_tokens,
            sorting_keys: ['tokens'],
            sampling_ratios: {
                'exemplar': exemplar_ratio,
                'full text': 1.0,
            }
        }
    },

    validation_data_loader: {
        batch_sampler: {
            type: "max_tokens_sampler",
            max_tokens: max_inference_tokens,
            sorting_keys: ['tokens']
        }
    },

    model: {
        type: "span",
        word_embedding: {
            token_embedders: {
                "pieces": {
                    type: "pretrained_transformer",
                    model_name: pretrained_model,
                }
            },
        },
        span_extractor: {
            type: 'combo',
            sub_extractors: [
                {
                    type: 'self_attentive',
                },
                {
                    type: 'bidirectional_endpoint',
                }
            ]
        },
        span_finder: {
            type: "bio",
            bio_encoder: {
                type: "lstm",
                hidden_size: bio_dim,
                num_layers: bio_layers,
                bidirectional: true,
                dropout: dropout,
            },
            no_label: false,
        },
        span_typing: {
            type: 'mlp',
            hidden_dims: span_typing_dims,
        },
        metrics: [{type: "srl"}],

        typing_loss_factor: typing_loss_factor,
        ontology_path: ontology_path,
        label_dim: label_dim,
        max_decoding_spans: 128,
        max_recursion_depth: 2,
        debug: debug,
    },

    trainer: {
        num_epochs: 128,
        patience: patience,
        [if std.length(cuda_devices) == 1 then "cuda_device"]: cuda_devices[0],
        validation_metric: "+em_f",
        grad_norm: 10,
        grad_clipping: 10,
        num_gradient_accumulation_steps: grad_acc,
        optimizer: {
            type: "transformer",
            base: {
                type: "adam",
                lr: 1e-3,
            },
            embeddings_lr: 0.0,
            encoder_lr: 1e-5,
            pooler_lr: 1e-5,
            layer_fix: layer_fix,
        }
    },

    cuda_devices:: cuda_devices,
    [if std.length(cuda_devices) > 1 then "distributed"]: {
        "cuda_devices": cuda_devices
    },
    [if std.length(cuda_devices) == 1 then "evaluate_on_test"]: true
}
