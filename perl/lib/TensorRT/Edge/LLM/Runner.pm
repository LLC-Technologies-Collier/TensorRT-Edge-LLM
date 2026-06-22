package TensorRT::Edge::LLM::Runner;

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

has multimodal_dir => (
    is       => 'ro',
    isa      => Str,
    default  => '',
);

has enable_cuda_graph => (
    is      => 'ro',
    default => 1,
);

has _xs_runtime => (
    is       => 'rw',
    isa      => Any,
    lazy     => 1,
    builder  => '_build_xs_runtime',
);

sub _build_xs_runtime {
    my $self = shift;
    $log->info("Initializing C++ LLMInferenceRuntime for " . $self->engine_dir);
    my $rt = $self->_xs_init_runtime($self->engine_dir, $self->multimodal_dir, $self->enable_cuda_graph);
    croak "Failed to initialize C++ LLMInferenceRuntime for " . $self->engine_dir unless defined $rt;
    return $rt;
}

sub DEMOLISH {
    my $self = shift;
    if (defined $self->{_xs_runtime}) {
        $log->debug("Destroying C++ LLMInferenceRuntime");
        $self->_xs_destroy_runtime($self->{_xs_runtime});
    }
}

sub generate {
    my ($self, $prompt, %options) = @_;
    
    $log->debugf("Generating response for prompt: %s (options: %s)", $prompt, \%options);
    
    my $max_gen_len = $options{max_tokens}   // 128;
    my $temperature = $options{temperature} // 1.0;
    my $top_p       = $options{top_p}       // 1.0;
    my $top_k       = $options{top_k}       // 1;
    my $seed        = $options{seed}        // int(rand(2**32));

    my $response = $self->_xs_generate(
        $self->_xs_runtime,
        $prompt, 
        $max_gen_len,
        $temperature,
        $top_p,
        $top_k,
        $seed
    );
    
    unless (defined $response) {
        croak "LLM generation failed";
    }
    
    return $response;
}

sub get_notify_fd {
    my $self = shift;
    return $self->_xs_get_notify_fd($self->_xs_runtime);
}

sub generate_async {
    my ($self, $prompt, %options) = @_;
    
    my $max_gen_len = $options{max_tokens}   // 128;
    my $temperature = $options{temperature} // 1.0;
    my $top_p       = $options{top_p}       // 1.0;
    my $top_k       = $options{top_k}       // 1;
    my $seed        = $options{seed}        // int(rand(2**32));
    my $stream      = $options{stream}      // 0;

    return $self->_xs_generate_async(
        $self->_xs_runtime,
        $prompt, 
        $max_gen_len,
        $temperature,
        $top_p,
        $top_k,
        $seed,
        $stream
    );
}

sub is_job_done {
    my ($self, $job_id) = @_;
    return $self->_xs_is_job_done($self->_xs_runtime, $job_id);
}

sub poll_tokens {
    my ($self, $job_id) = @_;
    return $self->_xs_poll_tokens($self->_xs_runtime, $job_id);
}

sub tokenize {
    my ($self, $request, $apply_chat_template) = @_;
    $apply_chat_template //= 1;
    return $self->_xs_tokenize($self->_xs_runtime, $request, $apply_chat_template);
}

sub decode {
    my ($self, $token_ids) = @_;
    return $self->_xs_decode($self->_xs_runtime, $token_ids);
}

sub collect_job {
    my ($self, $job_id) = @_;
    return $self->_xs_collect_job($self->_xs_runtime, $job_id);
}

sub create_session {
    my $self = shift;
    return $self->_xs_create_session($self->_xs_runtime);
}

sub destroy_session {
    my ($self, $session) = @_;
    $self->_xs_destroy_session($self->_xs_runtime, $session);
}

sub session_push_content {
    my ($self, $session, $content) = @_;
    $self->_xs_session_push_content($self->_xs_runtime, $session, $content);
}

sub session_poll_tokens {
    my ($self, $session) = @_;
    return $self->_xs_session_poll_tokens($session);
}

sub session_is_done {
    my ($self, $session) = @_;
    return $self->_xs_session_is_done($session);
}

sub release_spare_memory {
    my $self = shift;
    $log->debug("Releasing spare memory from C++ runtime");
    return $self->_xs_release_spare_memory($self->_xs_runtime);
}

sub get_embedding {
    my ($self, $text) = @_;
    return $self->_xs_get_embedding($self->_xs_runtime, $text);
}

1;