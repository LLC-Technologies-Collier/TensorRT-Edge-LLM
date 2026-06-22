#!/usr/bin/env perl

use strict;
use warnings;
use Test::More;
use Test::Exception;
use Test::MockModule;
use Log::Any::Test;
use Log::Any qw($log);
use Log::Any::Adapter;
use C9h::Test::Log qw(check_expected_logs);

Log::Any::Adapter->set('Test', min_level => 'trace');

use TensorRT::Edge::LLM::Runner;
use Path::Tiny;

subtest 'Constructor' => sub {
    my $temp_dir = Path::Tiny->tempdir();
    $temp_dir->child('config.json')->spew_utf8('{"builder_config": {"max_batch_size": 1, "max_input_len": 128, "max_output_len": 128, "max_kv_cache_capacity": 128, "max_lora_rank": 64, "eagle_base": false, "is_vlm": false}, "num_hidden_layers": 1, "num_key_value_heads": 1, "num_attention_heads": 1, "head_dim": 128, "hidden_size": 1024, "vocab_size": 32000}'); # Create dummy config
    $temp_dir->child('llm.engine')->spew_utf8('DUMMY ENGINE');
    my $runner = TensorRT::Edge::LLM::Runner->new(engine_dir => $temp_dir->stringify);
    isa_ok($runner, 'TensorRT::Edge::LLM::Runner');
    is($runner->engine_dir, $temp_dir->stringify, 'engine_dir stored');
    done_testing();
};

subtest 'generate success' => sub {
    $log->clear();
    
    my $mock_module = Test::MockModule->new('TensorRT::Edge::LLM::Runner');
    $mock_module->mock('_xs_init_runtime', sub {
        return "MOCK_RUNTIME_PTR";
    });
    $mock_module->mock('_xs_destroy_runtime', sub { 1 });
    $mock_module->mock('_xs_generate', sub {
        my ($self, $ctx, $prompt) = @_;
        return "The sky is blue.";
    });

    my $temp_dir = Path::Tiny->tempdir();
    $temp_dir->child('config.json')->spew_utf8('{"builder_config": {"max_batch_size": 1, "max_input_len": 128, "max_output_len": 128, "max_kv_cache_capacity": 128, "max_lora_rank": 64, "eagle_base": false, "is_vlm": false}, "num_hidden_layers": 1, "num_key_value_heads": 1, "num_attention_heads": 1, "head_dim": 128, "hidden_size": 1024, "vocab_size": 32000}'); # Create dummy config
    $temp_dir->child('llm.engine')->spew_utf8('DUMMY ENGINE');
    my $runner = TensorRT::Edge::LLM::Runner->new(engine_dir => $temp_dir->stringify);
    my $response = $runner->generate("Why is the sky blue?");

    is($response, "The sky is blue.", 'Correct response returned');
    
    my @expected_logs = (
        { level => 'info',  category => 'TensorRT::Edge::LLM::Runner', regex => qr/Initializing C\+\+ LLMInferenceRuntime/ },
        { level => 'debug', category => 'TensorRT::Edge::LLM::Runner', regex => qr/Generating response/ },
    );
    check_expected_logs(\@expected_logs, $log, 'Logs match success flow');

    done_testing();
};

done_testing();
