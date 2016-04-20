"""Utilities for manipulating labels from the Radio Galaxy Zoo."""

from . import config

def make_radio_combination_signature(radio_annotation):
    """Generates a unique signature for a radio annotation.
    
    radio_annotation: 'radio' dictionary from a classification.
    -> Something immutable
    """
    # My choice of immutable object will be a tuple of the xmax values,
    # sorted to ensure determinism, and rounded to nix floating point errors.
    # Note that the x scale is not the same as the IR scale, but the scale factor is
    # included in the annotation, so I have multiplied this out here for consistency.
    # Sometimes, there's no scale, so I've included a default scale.
    xmaxes = [round(float(c['xmax']) *
                    float(c.get('scale_width', '2.1144278606965172')), 14)
              for c in radio_annotation.values()]
    return tuple(sorted(xmaxes))

def parse_classification(classification):
    """Converts a raw RGZ classification into a classification dict.

    classification: RGZ classification dict.
    -> dict mapping radio signature to corresponding IR host pixel location.
    """
    result = {}

    for annotation in classification['annotations']:
        if 'radio' not in annotation:
            # This is a metadata annotation and we can ignore it.
            continue
        
        radio_signature = make_radio_combination_signature(annotation['radio'])
        
        if annotation['ir'] == 'No Sources':
            ir_location = None
        else:
            ir_location = (float(annotation['ir']['0']['x']),
                           float(annotation['ir']['0']['y']))

        result[radio_signature] = ir_location

    return result

