.. include:: macros.hrst

Elastic Class
-------------

.. autoclass:: physical_learning.elastic_utils.Elastic

	{% block attributes %}
	{% if attributes %}
	.. rubric:: Attributes

	.. autosummary::
		:toctree: {{ physical_learning.elastic_utils.Elastic }}
	{% for item in attributes %}
		~{{ name }}.{{ item }}
	{%- endfor %}
	{% endif %}
	{% endblock %}

    :members:
    :show-inheritance:

