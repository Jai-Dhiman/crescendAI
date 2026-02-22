fn main() {
    #[cfg(feature = "uniffi")]
    uniffi::generate_scaffolding("src/crescendai_shared.udl").unwrap();
}
