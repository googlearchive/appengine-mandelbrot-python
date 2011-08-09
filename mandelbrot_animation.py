#!/usr/bin/env python
#
# Copyright 2011 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Displays mandelbrot animations using PIL and NumPy."""

import functools
import itertools
import struct

try:
    # Use futures to generate mandelbrot images in parallel.
    from concurrent import futures
except ImportError:
    # If futures aren't available then gracefully fall back to using sequential
    # mandelbrot image generation.
    futures = None

import numpy
from PIL import GifImagePlugin
from PIL import Image
import webapp2


def build_animated_gif(stream, images, delay=0.1):
    """Writes an animated GIF into a stream given an iterator of PIL.Images.

    See http://en.wikipedia.org/wiki/Graphics_Interchange_Format#Animated_GIF.
    """
    # Process the images lazily to avoid unnecessary memory load. The first
    # image is used to build the GIF headers.
    images = (image.convert('P') for image in images)
    image = images.next()

    # Header
    stream.write("GIF89a")

    # Logical Screen Descriptor
    stream.write(struct.pack('<H', image.size[0]))
    stream.write(struct.pack('<H', image.size[1]))
    stream.write("\x87\x00\x00")

    # Palette
    stream.write(GifImagePlugin.getheader(image)[1])

    # Application Extension
    stream.write("\x21\xFF\x0B")
    stream.write("NETSCAPE2.0")
    stream.write("\x03\x01")
    stream.write(struct.pack('<H', 2**16-1))
    stream.write('\x00')

    for im in itertools.chain([image], images):
        # Graphic Control Extension
        stream.write('\x21\xF9\x04')
        stream.write('\x08')
        stream.write(struct.pack('<H', int(round(delay*100))))
        stream.write('\x00\x00')

        data = GifImagePlugin.getdata(im)
        for d in data:
            stream.write(d)

    # GIF file terminator
    stream.write(";")


def draw_mandelbrot(width, height,
                    left, right, top, bottom,
                    iterations,
                    z0):
    """Returns a PIL.Image representing the given portion of the set."""
    x, y = numpy.meshgrid(numpy.linspace(left, right, width),
                          numpy.linspace(top, bottom, height))
    c = x + 1j*y
    z = c.copy()
    z[:] = z0

    for n in range(iterations):
        z *= z
        z += c
    fractal = 255 * numpy.floor(
        iterations - numpy.log2(numpy.log10(1+abs(z)))) / iterations
    return Image.fromarray(fractal)


def generate_mandelbrot_animation(
        width, height, left, right, top, bottom, iterations,
        start_z0, end_z0, step_z0):
    """Yields PIL.Images representing the given mandelbrot animation."""
    z0s = numpy.arange(start_z0, end_z0 + step_z0, step_z0)
    fn = functools.partial(draw_mandelbrot,
                           width, height, left, right, top, bottom,
                           iterations)

    if futures:
        with futures.ThreadPoolExecutor(max_workers=5) as executor:
            # ThreadPoolExecutor has a bug where work items are not submitted
            # until the first result is requested. So yield each image
            # explicitly to prevent work from being submitted after the
            # with block falls-out and the executor is shutdown.
            for image in executor.map(fn, z0s):
                yield image
    else:
        for image in itertools.imap(fn, z0s):
            yield image


class MandelbrotHandler(webapp2.RequestHandler):
    "Respond to GET requests with an animated mandelbrot GIF."

    def get(self):
        left = float(self.request.get('left', '-2.68'))
        right = float(self.request.get('right', '1.32'))
        top = float(self.request.get('top', '-1.5'))
        bottom = float(self.request.get('bottom', '1.5'))
        iterations = int(self.request.get('iterations', '10'))

        width = int(self.request.get('width', '250'))
        height = int(self.request.get('height', '250'))

        start_z0 = float(self.request.get('start_z0', '-3'))
        end_z0 = float(self.request.get('end_z0', '3'))
        step_z0 = float(self.request.get('step_z0', '0.1'))

        images = generate_mandelbrot_animation(
                width, height, left, right, top, bottom, iterations,
                start_z0, end_z0, step_z0)
        self.response.headers['Content-Type'] = 'image/gif'
        build_animated_gif(self.response.out, images)


application = webapp2.WSGIApplication([(r'/.*', MandelbrotHandler)], debug=True)
