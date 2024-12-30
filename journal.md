# Journals

### Dec 2
- Added support for rendering for the colliders both Dirichlet and SDF
- [TODO] the plane rendering only support normal `(0,1,0)`
- [TODO] currently the SDF loaded is always a sphere. we need to make the render function a bit more better to account for SDFs other than spheres. the `SDF_collider` cannot store an SDF as it is a `wp.struct`

### Dec 6
- [x]  Rearrange the scene so the particle falls on the collider

### Dec 7
- [x] Sphere collision works!
- [x] Setup the scene with the rocks loaded as the SDF and make sure the rendering works
- [x] Write the collision detection function for the rocks
- [x] Get  the collider to move as a function of time

## December 22
- [ ] Realizing that collider traj does not actually store tranformations but actual positions. need to change the storing and accesssing

## December 24
- [x] Currently going to implicitly apply positions if that works, if not (DID NOT WORK)
- [x] going to use a function of velocity to see if that works better (SORT OF WORKS)
- [ ] Eventually The Gaussians should be used to create a gaussian field and that fields should figure out the velocity of all the surface points (convex hull) for the template mesh. makes you better leverage the gaussians too!


## December 30
- [ ] Need to discuss if we can assume the collider shape and trajectory to be known to more strongly enforce the boundary conditions using that too. This was done in the ultrasound thesis Shaifali shared
