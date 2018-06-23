import yaml


class Serializer():

    _serializable = []

    def to_dict(self):
        d = {}
        for key in self.__class__._serializable:
            if isinstance(key, tuple):
                value = getattr(self, key[0])
                d[key[0]] = {}
                for k, v in value.items():
                    d[key[0]][k] = v.to_dict()
            else:
                value = getattr(self, key)
                if isinstance(value, Serializer):
                    d[key] = value.to_dict()
                else:
                    d[key] = value
        return d

    def from_dict(self, obj_dict):
        for key, value in obj_dict.items():
            true_key = self.get_key(key)

            if isinstance(true_key, type):
                d = {}
                setattr(self, key, d)
                if value:
                    for k, v in value.items():
                        attr = true_key()
                        attr.from_dict(v)
                        d[k] = attr
            else:
                if isinstance(getattr(self, key), Serializer):
                    getattr(self, key).from_dict(value)
                else:
                    setattr(self, key, value)

    def get_key(self, key):
        for class_key in self.__class__._serializable:
            if isinstance(class_key, tuple):
                if class_key[0] == key:
                    return class_key[1]
            else:
                if class_key == key:
                    return class_key

    def to_yaml(self, yaml_path=None):
        yaml_string = yaml.dump(self.to_dict(), default_flow_style=False)
        if yaml_path:
            with open(yaml_path, "w") as f:
                f.write(yaml_string)
        else:
            return yaml_string

    def from_yaml(self, yaml_path):
        with open(yaml_path) as f:
            yaml_dict = yaml.load(f)
        self.from_dict(yaml_dict)

    def __str__(self):
        return self.to_yaml()

    def __repr__(self):
        return self.__str__()

    def _validate(self):
        for key in self.__class__._serializable:
            if isinstance(key, tuple):
                key = key[0]
                value = getattr(self, key)
                for k, v in value.items():
                    v._validate()
            else:
                value = getattr(self, key)
                if isinstance(value, Serializer):
                    value._validate()


def from_dict(dict):
    pass