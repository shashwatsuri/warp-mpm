# Journals

### Dec 2
- Added support for rendering for the colliders both Dirichlet and SDF
- [TODO] the plane rendering only support normal `(0,1,0)`
- [TODO] currently the SDF loaded is always a sphere. we need to make the render function a bit more better to account for SDFs other than spheres. the `SDF_collider` cannot store an SDF as it is a `wp.struct`

### Dec 6
- [x]  Rearrange the scene so the particle falls on the collider

### Dec 7
- [x] Sphere collision works!
- [ ] Setup the scene with the rocks loaded as the SDF and make sure the rendering works
- [ ] Write the collision detection function for the rocks
- [ ] Get  the collider to move as a function of time