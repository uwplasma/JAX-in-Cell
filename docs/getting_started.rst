Getting Started
===============

Prerequisites
-------------

- **Programming Language:** Python  
- Libraries: ``jax``, ``jax_tqdm``, ``matplotlib``

Installation
------------

**Using PyPi:**

.. code-block:: console

    pip install jaxincell

**Build from source:**

.. code-block:: console

    git clone https://github.com/uwplasma/JAX-in-Cell
    cd JAX-in-Cell
    pip install -r /path/to/requirements.txt
    pip install -e .

Usage
-----

.. code-block:: console

    jaxincell

To run using an input TOML file:

.. code-block:: console

    jaxincell example_input.toml

Run example script:

.. code-block:: console

    python example_script.py

Testing
-------

.. code-block:: console

    pytest .
