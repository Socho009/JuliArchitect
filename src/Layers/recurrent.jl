# ─── Recurrent Cells: RNN, LSTM, GRU ──────────────────────────────────────────
#
# Architect-native reimagining of Karpathy's char-rnn architecture.
# Each cell implements a single-timestep forward pass dispatched on cell type.
# Uses the existing Linear{P} layer for all weight matrices.
#
# Mathematical reference:
#   RNN:  h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b)
#   LSTM: Hochreiter & Schmidhuber (1997), fused gate formulation
#   GRU:  Cho et al. (2014), update-reset gate formulation

# ─── Sigmoid Utility ──────────────────────────────────────────────────────────

"""
    _sigmoid(x::P) where P

Logistic sigmoid activation: σ(x) = 1 / (1 + exp(-x)).
"""
_sigmoid(x::P) where {P<:AbstractFloat} = one(P) / (one(P) + exp(-x))

# ─── Recurrent State Types ────────────────────────────────────────────────────

"""
    RNNState{P, M<:AbstractMatrix{P}} <: AbstractRecurrentState{P}

Hidden state container for vanilla RNN and GRU cells.

# Fields
- `h::M` — Hidden state of shape (hidden_size, batch)
"""
struct RNNState{P, M<:AbstractMatrix{P}} <: AbstractRecurrentState{P}
    h::M
end

"""
    LSTMState{P, M<:AbstractMatrix{P}} <: AbstractRecurrentState{P}

Dual state container for LSTM cells. The cell state `c` enables long-range
gradient flow through additive interactions (no squashing between timesteps).

# Fields
- `c::M` — Cell state of shape (hidden_size, batch)
- `h::M` — Hidden state of shape (hidden_size, batch)
"""
struct LSTMState{P, M<:AbstractMatrix{P}} <: AbstractRecurrentState{P}
    c::M
    h::M
end

# ─── Vanilla RNN Cell ─────────────────────────────────────────────────────────

"""
    RNNCell{P, L<:Linear{P}} <: AbstractRecurrentCell{P}

Elman RNN cell: h_t = tanh(W_ih·x_t + b_ih + W_hh·h_{t-1} + b_hh)

The simplest recurrent unit. Suffers from vanishing/exploding gradients
for long sequences, but serves as the foundation for LSTM and GRU.

# Fields
- `W_ih::L` — Input-to-hidden projection (hidden_size, input_size)
- `W_hh::L` — Hidden-to-hidden projection (hidden_size, hidden_size)
- `hidden_size::Int`
"""
struct RNNCell{P, L<:Linear{P}} <: AbstractRecurrentCell{P}
    W_ih::L
    W_hh::L
    hidden_size::Int
end

"""
    RNNCell{P}(input_size::Int, hidden_size::Int) where P

Construct an RNN cell with given input and hidden dimensions.
"""
function RNNCell{P}(input_size::Int, hidden_size::Int) where {P<:AbstractFloat}
    W_ih = Linear{P}(input_size, hidden_size; bias=true)
    W_hh = Linear{P}(hidden_size, hidden_size; bias=true)
    RNNCell{P, typeof(W_ih)}(W_ih, W_hh, hidden_size)
end

"""
    init_state(cell::RNNCell{P}, batch_size::Int) where P

Create zero-initialized hidden state for the RNN cell.
"""
function init_state(cell::RNNCell{P}, batch_size::Int) where P
    h = zeros(P, cell.hidden_size, batch_size)
    RNNState{P, typeof(h)}(h)
end

"""
    (cell::RNNCell{P})(x::AbstractMatrix{P}, state::RNNState{P}) where P

Single-timestep RNN forward pass.
Input x: (input_size, batch), state.h: (hidden_size, batch)
Returns: (h_new, new_state) where h_new is (hidden_size, batch)
"""
function (cell::RNNCell{P})(x::AbstractMatrix{P}, state::RNNState{P}) where P
    h_new = tanh.(cell.W_ih(x) .+ cell.W_hh(state.h))
    return h_new, RNNState{P, typeof(h_new)}(h_new)
end

function Base.show(io::IO, cell::RNNCell{P}) where P
    in_dim = size(cell.W_ih.weight, 2)
    print(io, "RNNCell{$P}($in_dim → $(cell.hidden_size))")
end

# ─── LSTM Cell ─────────────────────────────────────────────────────────────────

"""
    LSTMCell{P, L<:Linear{P}} <: AbstractRecurrentCell{P}

Long Short-Term Memory cell with fused gate computation (Hochreiter & Schmidhuber, 1997).

Computes all four gates in a single matmul for efficiency:
    [i; f; o; g] = [σ; σ; σ; tanh](W_ih·x + W_hh·h + b)
    c_t = f ⊙ c_{t-1} + i ⊙ g
    h_t = o ⊙ tanh(c_t)

The cell state c flows through time with only additive interactions,
enabling long-range gradient flow. Forget gate bias is initialized to 1.0
(Jozefowicz et al., 2015) to encourage remembering by default.

# Fields
- `W_ih::L` — Input-to-gates projection (4*hidden_size, input_size)
- `W_hh::L` — Hidden-to-gates projection (4*hidden_size, hidden_size)
- `hidden_size::Int`
"""
struct LSTMCell{P, L<:Linear{P}} <: AbstractRecurrentCell{P}
    W_ih::L
    W_hh::L
    hidden_size::Int
end

"""
    LSTMCell{P}(input_size::Int, hidden_size::Int) where P

Construct an LSTM cell with fused gate projections.
"""
function LSTMCell{P}(input_size::Int, hidden_size::Int) where {P<:AbstractFloat}
    W_ih = Linear{P}(input_size, 4 * hidden_size; bias=true)
    W_hh = Linear{P}(hidden_size, 4 * hidden_size; bias=true)
    LSTMCell{P, typeof(W_ih)}(W_ih, W_hh, hidden_size)
end

"""
    init_state(cell::LSTMCell{P}, batch_size::Int) where P

Create zero-initialized cell and hidden states for the LSTM cell.
"""
function init_state(cell::LSTMCell{P}, batch_size::Int) where P
    c = zeros(P, cell.hidden_size, batch_size)
    h = zeros(P, cell.hidden_size, batch_size)
    LSTMState{P, typeof(h)}(c, h)
end

"""
    (cell::LSTMCell{P})(x::AbstractMatrix{P}, state::LSTMState{P}) where P

Single-timestep LSTM forward pass with fused gate computation.
Input x: (input_size, batch), state: (c, h) each (hidden_size, batch)
Returns: (h_new, new_state)

Gate layout: [input_gate; forget_gate; output_gate; cell_candidate]
"""
function (cell::LSTMCell{P})(x::AbstractMatrix{P}, state::LSTMState{P}) where P
    H = cell.hidden_size

    # Fused gate computation: (4H, batch)
    gates = cell.W_ih(x) .+ cell.W_hh(state.h)

    # Split and activate gates
    i_gate = _sigmoid.(gates[1:H, :])           # input gate
    f_gate = _sigmoid.(gates[H+1:2H, :])        # forget gate
    o_gate = _sigmoid.(gates[2H+1:3H, :])       # output gate
    g_gate = tanh.(gates[3H+1:4H, :])           # cell candidate

    # LSTM update equations
    c_new = f_gate .* state.c .+ i_gate .* g_gate
    h_new = o_gate .* tanh.(c_new)

    return h_new, LSTMState{P, typeof(h_new)}(c_new, h_new)
end

function Base.show(io::IO, cell::LSTMCell{P}) where P
    in_dim = size(cell.W_ih.weight, 2)
    print(io, "LSTMCell{$P}($in_dim → $(cell.hidden_size))")
end

# ─── GRU Cell ──────────────────────────────────────────────────────────────────

"""
    GRUCell{P, L<:Linear{P}} <: AbstractRecurrentCell{P}

Gated Recurrent Unit cell (Cho et al., 2014).

    z_t = σ(W_iz·x + W_hz·h + b_z)                 (update gate)
    r_t = σ(W_ir·x + W_hr·h + b_r)                 (reset gate)
    ĥ_t = tanh(W_in·x + W_hn·(r_t ⊙ h) + b_n)     (candidate)
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ĥ_t          (interpolation)

The update gate z performs learned interpolation: when z→0, the state
is copied unchanged (identity skip-connection through time).

# Fields
- `W_iz, W_hz` — Update gate projections
- `W_ir, W_hr` — Reset gate projections
- `W_in, W_hn` — Candidate hidden state projections
- `hidden_size::Int`
"""
struct GRUCell{P, L<:Linear{P}} <: AbstractRecurrentCell{P}
    W_iz::L;  W_hz::L    # update gate
    W_ir::L;  W_hr::L    # reset gate
    W_in::L;  W_hn::L    # candidate
    hidden_size::Int
end

"""
    GRUCell{P}(input_size::Int, hidden_size::Int) where P

Construct a GRU cell with separate gate projections.
"""
function GRUCell{P}(input_size::Int, hidden_size::Int) where {P<:AbstractFloat}
    W_iz = Linear{P}(input_size, hidden_size; bias=true)
    W_hz = Linear{P}(hidden_size, hidden_size; bias=true)
    W_ir = Linear{P}(input_size, hidden_size; bias=true)
    W_hr = Linear{P}(hidden_size, hidden_size; bias=true)
    W_in = Linear{P}(input_size, hidden_size; bias=true)
    W_hn = Linear{P}(hidden_size, hidden_size; bias=true)
    GRUCell{P, typeof(W_iz)}(W_iz, W_hz, W_ir, W_hr, W_in, W_hn, hidden_size)
end

"""
    init_state(cell::GRUCell{P}, batch_size::Int) where P

Create zero-initialized hidden state for the GRU cell.
"""
function init_state(cell::GRUCell{P}, batch_size::Int) where P
    h = zeros(P, cell.hidden_size, batch_size)
    RNNState{P, typeof(h)}(h)  # GRU uses same state shape as vanilla RNN
end

"""
    (cell::GRUCell{P})(x::AbstractMatrix{P}, state::RNNState{P}) where P

Single-timestep GRU forward pass.
Input x: (input_size, batch), state.h: (hidden_size, batch)
Returns: (h_new, new_state)
"""
function (cell::GRUCell{P})(x::AbstractMatrix{P}, state::RNNState{P}) where P
    # Gate computations
    z = _sigmoid.(cell.W_iz(x) .+ cell.W_hz(state.h))          # update gate
    r = _sigmoid.(cell.W_ir(x) .+ cell.W_hr(state.h))          # reset gate
    h_candidate = tanh.(cell.W_in(x) .+ cell.W_hn(r .* state.h))  # candidate

    # Interpolation: z=0 → copy old state, z=1 → use candidate
    h_new = (one(P) .- z) .* state.h .+ z .* h_candidate

    return h_new, RNNState{P, typeof(h_new)}(h_new)
end

function Base.show(io::IO, cell::GRUCell{P}) where P
    in_dim = size(cell.W_iz.weight, 2)
    print(io, "GRUCell{$P}($in_dim → $(cell.hidden_size))")
end

# Device transfer for recurrent cells
import ..Core

function Core.to_device(state::RNNState{P, M}, device::Symbol) where {P, M<:AbstractMatrix{P}}
    return RNNState{P, typeof(to_device(state.h, device))}(to_device(state.h, device))
end

function Core.to_device(state::LSTMState{P, M}, device::Symbol) where {P, M<:AbstractMatrix{P}}
    return LSTMState{P, typeof(to_device(state.c, device))}(to_device(state.c, device), to_device(state.h, device))
end

function Core.to_device(cell::RNNCell{P, L}, device::Symbol) where {P, L<:Linear{P}}
    W_ih_new = to_device(cell.W_ih, device)
    W_hh_new = to_device(cell.W_hh, device)
    return RNNCell{P, typeof(W_ih_new)}(W_ih_new, W_hh_new, cell.hidden_size)
end

function Core.to_device(cell::LSTMCell{P, L}, device::Symbol) where {P, L<:Linear{P}}
    W_ih_new = to_device(cell.W_ih, device)
    W_hh_new = to_device(cell.W_hh, device)
    return LSTMCell{P, typeof(W_ih_new)}(W_ih_new, W_hh_new, cell.hidden_size)
end

function Core.to_device(cell::GRUCell{P, L}, device::Symbol) where {P, L<:Linear{P}}
    W_iz_new = to_device(cell.W_iz, device)
    W_hz_new = to_device(cell.W_hz, device)
    W_ir_new = to_device(cell.W_ir, device)
    W_hr_new = to_device(cell.W_hr, device)
    W_in_new = to_device(cell.W_in, device)
    W_hn_new = to_device(cell.W_hn, device)
    return GRUCell{P, typeof(W_iz_new)}(W_iz_new, W_hz_new, W_ir_new, W_hr_new, W_in_new, W_hn_new, cell.hidden_size)
end
