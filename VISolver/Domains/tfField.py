class DNNField(object):

  def __init__(self, input_size, num_units, hidden_units=[], activation=tanh):
    self._input_size = input_size
    self._num_units = num_units
    self._hidden_units = hidden_units
    self._units = [input_size] + hidden_units + [input_size]
    self._activation = activation

  def dnn_fieldA(self,inputs):
    units = [self._input_size] + self._hidden_units + [self._input_size]

    fields = []
    for f in range(self._num_units):
      prev = inputs
      for i, unit in enumerate(units[:-1]):
        with tf.variable_scope('layer_'+str(i)):
          # Wi = tf.Variable(tf.random_normal([unit,units[i+1]]))
          # bi = tf.Variable(tf.zeros([units[i+1]]))
          # # if i == len(self._units) - 2:
          # #   hiddeni = tf.matmul(prev,Wi)+bi
          # # else:
          # hiddeni = self._activation(tf.matmul(prev,Wi)+bi)
          hiddeni = self._activation(_linear(args=prev,output_size=units[i+1],bias=True))
          prev = hiddeni
      fields += [hiddeni]

    return tf.pack(fields)

  def dnn_field2A(self,inputs,scope):
    # hidden = [input_size,5<input_size]
    units = [self._input_size] + self._hidden_units + [self._num_units*self._input_size]

    prev = inputs
    for i, unit in enumerate(units[:-1]):
      # Wi = tf.Variable(tf.random_normal([unit,units[i+1]]))
      # bi = tf.Variable(tf.zeros([units[i+1]]))
      # # if i == len(self._units) - 2:
      # #   hiddeni = tf.matmul(prev,Wi)+bi
      # # else:
      # hiddeni = self._activation(tf.matmul(prev,Wi)+bi)
      hiddeni = self._activation(_linear(args=prev,output_size=units[i+1],bias=True,scope=scope))
      prev = hiddeni

    fields = tf.reshape(hiddeni,shape=[-1,self._num_units,self._input_size])
    fields = tf.transpose(fields,[1,0,2])

    return fields

  def dnn_field(self,inputs):
    units = [self._input_size] + self._hidden_units + [self._input_size]

    fields = []
    for f in range(self._num_units):
      prev = inputs
      for i, unit in enumerate(units[:-1]):
        Wi = tf.Variable(tf.random_normal([unit,units[i+1]]))
        bi = tf.Variable(tf.zeros([units[i+1]]))
        # if i == len(self._units) - 2:
        #   hiddeni = tf.matmul(prev,Wi)+bi
        # else:
        hiddeni = self._activation(tf.matmul(prev,Wi)+bi)
        prev = hiddeni
      fields += [hiddeni]

    return tf.pack(fields)

  def dnn_field2(self,inputs):
    # hidden = [input_size,5<input_size]
    units = [self._input_size] + self._hidden_units + [self._num_units*self._input_size]

    prev = inputs
    for i, unit in enumerate(units[:-1]):
      Wi = tf.Variable(tf.random_normal([unit,units[i+1]]))
      bi = tf.Variable(tf.zeros([units[i+1]]))
      # if i == len(self._units) - 2:
      #   hiddeni = tf.matmul(prev,Wi)+bi
      # else:
      hiddeni = self._activation(tf.matmul(prev,Wi)+bi)
      prev = hiddeni

    fields = tf.reshape(hiddeni,shape=[-1,self._num_units,self._input_size])
    fields = tf.transpose(fields,[1,0,2])

    return fields

_FieldStateTuple = collections.namedtuple("FieldStateTuple", ("prev_inputs", "h"))

class FieldStateTuple(_FieldStateTuple):
  """Tuple used by Field Cells for `state_size`, `zero_state`, and output state.
  Stores two elements: `(prev_inputs, h)`, in that order.
  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (prev_inputs, h) = self
    if not prev_inputs.dtype == h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(prev_inputs.dtype), str(h.dtype)))
    return prev_inputs.dtype


class BasicFieldCell(RNNCell):
  """Basic Field recurrent network cell.
  The implementation is based on: Ian Gemp
  """

  def __init__(self, num_units, input_size, fields,
               state_is_tuple=True):
    """Initialize the basic Field cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      fields: F(input), functions to compute vector fields.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `prev_input` and `state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    self._num_units = num_units
    self._fields = fields
    self._input_size = input_size
    self._state_is_tuple = state_is_tuple

  @property
  def state_size(self):
    return (FieldStateTuple(self._input_size, self._num_units)
            if self._state_is_tuple else self._input_size + self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Field cell."""
    with vs.variable_scope(scope or "basic_field_cell") as scope:
      if self._state_is_tuple:
        prev_inputs, h = state
      else:
        prev_inputs = array_ops.slice(state, [0, 0], [-1, self._input_size])
        h = array_ops.slice(state, [0, self._input_size], [-1, self._num_units])

      prev_fields = self._fields(prev_inputs)
      scope.reuse_variables()
      fields = self._fields(inputs)
      trapezoid_fields = 0.5 * (prev_fields + fields)
      d_inputs = inputs - prev_inputs
      path_int = tf.reduce_sum(tf.mul(trapezoid_fields,d_inputs),reduction_indices=-1)
      path_int = tf.transpose(path_int)
      new_h = path_int + h

      if self._state_is_tuple:
        new_state = FieldStateTuple(inputs, new_h)
      else:
        new_state = array_ops.concat(1, [inputs, new_h])
      return new_h, new_state
