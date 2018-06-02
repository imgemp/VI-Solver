from collections import OrderedDict
import tensorflow as tf
ds = tf.contrib.distributions
slim = tf.contrib.slim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from VISolver.Domain import Domain

from IPython import embed

params = dict(
    batch_size=1024,  # 512
    disc_learning_rate=1e-3,  # 1e-4
    gen_learning_rate=1e-3,
    z_dim=256,
    x_dim=2,
    findiff_step=1e-3,
    gamma=1e-2,
)

class GMGAN(Domain):
    def __init__(self, params=params, dyn='FCC'):
        tf.reset_default_graph()

        data = self.sample_mog(params['batch_size'])

        noise = ds.Normal(tf.zeros(params['z_dim']), 
                          tf.ones(params['z_dim'])).sample(params['batch_size'])
        # Construct generator and discriminator nets
        with slim.arg_scope([slim.fully_connected], weights_initializer=tf.orthogonal_initializer(gain=1.4)):
            samples = self.generator(noise, output_dim=params['x_dim'])
            real_score = self.discriminator(data)
            fake_score = self.discriminator(samples, reuse=True)
            
        # Saddle objective    
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score, labels=tf.ones_like(real_score)) +
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.zeros_like(fake_score)))

        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
        gen_shapes = [tuple(v.get_shape().as_list()) for v in gen_vars]
        disc_shapes = [tuple(v.get_shape().as_list()) for v in disc_vars]

        # Generator gradient
        g_opt = tf.train.GradientDescentOptimizer(learning_rate=params['gen_learning_rate'])
        g_grads = g_opt.compute_gradients(-loss, var_list=gen_vars)

        # Discriminator gradient
        d_opt = tf.train.GradientDescentOptimizer(learning_rate=params['disc_learning_rate'])
        d_grads = d_opt.compute_gradients(loss, var_list=disc_vars)

        # Squared Norm of Gradient: d/dx 1/2||F||^2 = J^T F
        grads_norm_sep = [tf.reduce_sum(g[0]**2) for g in g_grads+d_grads]
        grads_norm = 0.5*tf.reduce_sum(grads_norm_sep)

        # Gradient of Squared Norm
        JTF = tf.gradients(grads_norm, xs=gen_vars+disc_vars)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self.params = params
        self.data = data
        self.samples = samples
        self.gen_vars = gen_vars
        self.disc_vars = disc_vars
        self.gen_shapes = gen_shapes
        self.disc_shapes = disc_shapes
        self.Fg = g_grads
        self.Fd = d_grads
        self.JTF = JTF
        self.sess = sess
        self.findiff_step = params['findiff_step']
        self.gamma = params['gamma']
        self.dyn = dyn

        if dyn == 'FCC':
            self.F = self.FCC
        else:
            self.F = self._F

    def get_weights(self):
        weights = self.sess.run(self.gen_vars+self.disc_vars)
        return np.concatenate([v.flatten() for v in weights])

    def set_weights(self,weights):
        # reshape weight array and create assign ops
        ptr = 0
        assign_ops = []
        for var, shape in zip(self.gen_vars + self.disc_vars, self.gen_shapes + self.disc_shapes):
            val = weights[ptr:ptr+np.prod(shape)].reshape(shape)
            assign_ops += [tf.assign(var,val)]
            ptr += np.prod(shape)
        # overwrite weights
        self.sess.run(assign_ops)

    def _F(self,weights):
        self.set_weights(weights)
        return np.concatenate([gv[0].flatten() for gv in self.sess.run(self.Fg+self.Fd)])

    def FCC(self,weights):
        _F = self._F(weights)  # overwrite weights and compute F(xk)
        JTF = np.concatenate([g.flatten() for g in self.sess.run(self.JTF)])  # compute JTF(xk)
        _Fnext = self._F(weights-self.findiff_step*_F)  # overwrite weights and compute F(xk+1)
        JF = (_F-_Fnext)/self.findiff_step
        self.set_weights(weights)  # reset weights
        return _F  - self.gamma*0.5*(JF-JTF)

    def sample_mog(self, batch_size, n_mixture=8, std=0.01, radius=1.0):
        thetas = np.linspace(0, 2 * np.pi, n_mixture)
        xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
        cat = ds.Categorical(tf.zeros(n_mixture))
        comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
        data = ds.Mixture(cat, comps)
        return data.sample(batch_size)

    def generator(self, z, output_dim=2, n_hidden=128, n_layer=2):
        with tf.variable_scope("generator"):
            h = slim.stack(z, slim.fully_connected, [n_hidden] * n_layer, activation_fn=tf.nn.tanh)
            x = slim.fully_connected(h, output_dim, activation_fn=None)
        return x

    def discriminator(self, x, n_hidden=128, n_layer=2, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            h = slim.stack(x, slim.fully_connected, [n_hidden] * n_layer, activation_fn=tf.nn.tanh)
            log_d = slim.fully_connected(h, 1, activation_fn=None)
        return log_d

    def visualize_dist(self):
        xx, yy = self.sess.run([self.samples, self.data])
        fig = plt.figure(figsize=(5,5))
        plt.scatter(xx[:, 0], xx[:, 1], edgecolor='none')
        plt.scatter(yy[:, 0], yy[:, 1], c='g', edgecolor='none')
        plt.axis('off')
        ax = plt.gca()
        return fig, ax


if __name__ == '__main__':
    params['batch_size'] = 1024
    gmgan = GMGAN(params)
    embed()