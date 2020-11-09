cdef class LT:
    cdef:
        int    [:] sensor_ids_
        double [:] zbins_
    cdef double* get_values_(self, const double x, const double y, const int sensor_id)
