struct {{object.name}} {
  {% for type, name in object.members %}
  {{type}} {{name}};
  {% endfor %}

  {% for function in object.functions %}
  {{function.to_cpp}}
  {% endfor %}
};
