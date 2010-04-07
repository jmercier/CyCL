<%def name="makesection(name)">
#
#
#   ${name}
#
#
</%def>

<%def name="info_getter(name, fct_name, obj_type, info_type, return_types)">
    %for t in return_types:
cdef ${t} _${name}_${t}(${obj_type} obj, ${info_type} param_name):
    cdef size_t size
    %if t != "bytes":
    cdef ${t} result
    cdef cl_int errcode = ${fct_name}(obj, param_name, sizeof(${t}), &result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return result
    %else:
    cdef char result[256]
    cdef cl_int errcode = ${fct_name}(obj, param_name, 256 * sizeof(char), result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef bytes s = result[:size -1]
    return s
    %endif

    %endfor
</%def>
<%def name="properties_getter(obj_type, internal, properties_desc)">
%for ptype in properties_desc:
    %for pname, pdefine in properties_desc[ptype]:
    property ${pname}:
        def __get__(self):
        %if isinstance(pdefine, tuple):
<%ret_string = "" %>\
            %for i, define in enumerate(pdefine):
                cdef ${ptype} r_${i} = _get${obj_type}Info_${ptype}(self.${internal},
                                        ${define})
<%ret_string += "r_%d, " % i %>\
            %endfor
                return (${ret_string})
        %else:
            return _get${obj_type}Info_${ptype}(self.${internal},
                                        ${pdefine})
        %endif
    %endfor
%endfor
</%def>

<%def name="properties_getter2(fct_name, internal, properties_desc)">\
%for ptype in properties_desc:

%for pname, pdefine in properties_desc[ptype]:
    property ${pname}:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
    %if isinstance(pdefine, tuple):
<%ret_string = "" %>\
        %for i, define in enumerate(pdefine):
<%ret_string += "r_%d, " % i %>\
            cdef ${ptype} r_${i}
        %endfor
        %for i, define in enumerate(pdefine):
            errcode = ${fct_name}(self.${internal},
                                  ${define},
                                  sizeof(${ptype}), &r_${i}, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
        %endfor
            return (${ret_string})
    %else:
            cdef ${ptype} result
        %if ptype == "bytes":
            cdef char char_result[256]
            errcode = ${fct_name}(self.${internal},
                                  ${pdefine},
                                  256 * sizeof(char), char_result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            result = char_result[:size - 1]
        %else:
            errcode = ${fct_name}(self.${internal},
                                  ${pdefine},
                                  sizeof(${ptype}), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
        %endif
            return result
    %endif

%endfor
%endfor
</%def>

<%def name="properties_repr(attributes)">\
<%repr_str = "<%s"%>\
<%prop_str = "self.__class__.__name__,"%>\
%for pname in attributes:
<%repr_str += " " + pname + '="%s"' %>\
<%prop_str += " self." + pname + "," %>\
%endfor
<%repr_str += ">" %>\
def __repr__(self):
        return '${repr_str}' % <%text> \</%text>
                (${prop_str} )\
</%def>
<%def name="make_dealloc(command)">\
def __dealloc__(self):
        cdef cl_int errcode
        errcode = ${command}\
<%text>
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % \
                                self.__class__.__name__)
</%text>\
</%def>
