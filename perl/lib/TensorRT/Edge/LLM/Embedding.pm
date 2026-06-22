package TensorRT::Edge::LLM::Embedding;

use strict;
use warnings;
use Moo;
use Carp qw(croak);
use Log::Any qw($log);
use Types::Standard qw(Str Int Any);

our $VERSION = '0.01';
use XSLoader;
XSLoader::load(__PACKAGE__, $VERSION);

has engine_dir => (
    is       => 'ro',
    isa      => Str,
    required => 1,
);

has trt_llm_plugins => ( is => 'ro' );
has edgellm_plugins => ( is => 'ro' );

has _xs_runtime => (
    is       => 'rw',
    isa      => Any,
    lazy     => 1,
    builder  => '_build_xs_runtime',
);

sub init_plugins {
    my ($self, $path) = @_;
    return $self->_xs_init_plugins($path);
}

sub _build_xs_runtime {
    my $self = shift;
    $log->info("Initializing C++ LLMInferenceRuntime for Embedding: " . $self->engine_dir);
    
    # Initialize from explicit arguments or environment
    my @plugins = grep { defined } ($self->trt_llm_plugins, $self->edgellm_plugins);
    if (!@plugins && $ENV{EDGELLM_PLUGIN_PATH}) {
        push @plugins, $ENV{EDGELLM_PLUGIN_PATH};
    }
    
    $self->init_plugins($_) for @plugins;
    
    my $rt = $self->_xs_init_runtime($self->engine_dir);
    croak "Failed to initialize C++ LLMInferenceRuntime for " . $self->engine_dir unless defined $rt;
    return $rt;
}

sub DEMOLISH {
    my $self = shift;
    if (defined $self->{_xs_runtime}) {
        $self->_xs_destroy_runtime($self->{_xs_runtime});
    }
}

sub get_embedding {
    my ($self, $text) = @_;
    
    $log->debugf("Extracting embedding for: %s", $text);
    
    my $vector = $self->_xs_get_embedding($self->_xs_runtime, $text);
    croak "Failed to extract embedding" unless defined $vector;
    
    return $vector;
}

1;
