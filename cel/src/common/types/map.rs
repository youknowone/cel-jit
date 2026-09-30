use crate::common::traits;

pub(crate) fn stdlib(env: &mut crate::Env) {
    env.add_overload(
        "size",
        "size_map",
        vec![super::MAP_TYPE],
        traits::adapter::sizer_size,
    )
    .expect("Must be unique id");
    env.add_member_overload(
        "size",
        "map_size",
        super::MAP_TYPE,
        vec![],
        traits::adapter::sizer_size,
    )
    .expect("Must be unique id");
}
