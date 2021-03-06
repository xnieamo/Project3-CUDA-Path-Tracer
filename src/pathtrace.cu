#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define DIRECTLIGHTING 1
#define CACHEFIRSTRAY 1
#define ANTIALIAS 1
#define STREAMCOMPACT 1
#define MATERIALSORT 0 // Doesn't work for some reason, breaks on sort_by_key
#define TIME 0
#define LENSJITTER 0
#define BLOCKSIZE 128

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;

// For caching first bounce
static PathSegment * dev_first_cache = NULL;
static bool firstTime = true;

// For material sorting
static int * dev_materialIds = NULL;

// Direct lighting variable
static int numLights = 0;

void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_first_cache, pixelcount * sizeof(PathSegment));
	//cudaMalloc(&dev_shadowFeelers, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_materialIds, pixelcount * sizeof(int));
	cudaMemset(dev_materialIds, 0, pixelcount * sizeof(int));

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_first_cache);
	cudaFree(dev_materialIds);
	checkCUDAError("pathtraceFree");
}

__device__ __host__
void lensJitter(PathSegment & path, Camera & cam, float u, float v){
	concentricSampleDisc(u, v);

	u *= cam.lensRadius;
	v *= cam.lensRadius;

	float ft = cam.focalDistance / glm::abs(path.ray.direction[2]);
	glm::vec3 pFocus = path.ray.origin + ft * path.ray.direction;
	path.ray.origin += glm::vec3(u, v, 0.f);
	path.ray.direction = glm::normalize(pFocus - path.ray.origin);
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
#if DIRECTLIGHTING
		segment.color = glm::vec3(0.0f, 0.0f, 0.0f);
#else
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
#endif
		segment.throughput = glm::vec3(1.f);
		segment.inside = false;
	
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

#if ANTIALIAS
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x +  u01(rng) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
			);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);
#endif

#if LENSJITTER
		// Apply lens jitter for depth of field
		lensJitter(segment, cam, u01(rng), u01(rng));
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// Copied from body of computeIntersections to make it a bit more flexible
__device__ __host__ void computeSingleIntersection(
	PathSegment & pathSegment
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection & intersection) {

	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	bool outside = true;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	for (int i = 0; i < geoms_size; i++)
	{
		Geom & geom = geoms[i];

		if (geom.type == CUBE)
		{
			t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == SPHERE)
		{
			t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
		}
		// TODO: add more intersection tests here... triangle? metaball? CSG?

		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t_min > t)
		{
			t_min = t;
			hit_geom_index = i;
			intersect_point = tmp_intersect;
			normal = tmp_normal;
		}
	}

	if (hit_geom_index == -1)
	{
		intersection.t = -1.0f;
	}
	else
	{
		//The ray hits something
		intersection.t = t_min;
		intersection.materialId = geoms[hit_geom_index].materialid;
		intersection.surfaceNormal = normal;
		intersection.intersect = intersect_point;
		intersection.surfaceArea = geoms[hit_geom_index].surfaceArea;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		computeSingleIntersection(pathSegment, geoms, geoms_size, intersections[path_index]);
	}
}

// Use actual shader
__global__ void shadeMaterial(
	int iter
	, int depth
	, int num_paths
	, int num_lights
	, int num_geoms
	, Geom * geoms
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{

		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (pathSegments[idx].remainingBounces > 0 && intersection.t > 0.0f) { // if the intersection exists...
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at
			// makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.f) {
#if DIRECTLIGHTING
				if (depth == 0) {
					pathSegments[idx].color = pathSegments[idx].throughput * (material.color * material.emittance);
				}
#else
				pathSegments[idx].color *= (material.color * material.emittance);
#endif
				pathSegments[idx].remainingBounces = 0;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			else {

#if DIRECTLIGHTING	
				thrust::uniform_real_distribution<float> u0L(0, num_lights);

				// Pick a random light
				int chosenLight = (int)u0L(rng);
				Geom theLight = geoms[chosenLight];

				// Do some independent operations. I think almost everything
				// depends on this light, so just allocate some variables.
				glm::vec4 lp = glm::vec4(0.f, 0.f, 0.f, 1.f);
				PathSegment shadowFeeler;
				ShadeableIntersection shadowIntersect;
				ShadeableIntersection brdfIntersect;

				// Load pathsegment here so we save some time
				PathSegment brdfSample = pathSegments[idx];

				// Generate a ray pointing towards that light
				if (theLight.type == CUBE) {
					lp = glm::vec4(sampleCube(rng), 1.f);
				}
				else if (theLight.type == SPHERE) {
					lp = glm::vec4(sampleSphere(rng), 1.f);
				}

				glm::vec3 lightPos = glm::vec3(theLight.transform * lp);
				shadowFeeler.ray.direction = glm::normalize(lightPos - intersection.intersect);
				shadowFeeler.ray.origin = intersection.intersect + 0.01f * shadowFeeler.ray.direction;

				// This is our light intersection
				computeSingleIntersection(shadowFeeler, geoms, num_geoms, shadowIntersect);
			
				// Now we need to sample the brdf once and see if that hits the light
				sampleBrdf(brdfSample,
					intersection.intersect,
					intersection.surfaceNormal,
					material,
					rng);
				computeSingleIntersection(brdfSample, geoms, num_geoms, brdfIntersect);

				// Do direct lighting
				scatterWithDirectLighting(
					pathSegments[idx],
					shadowFeeler,
					brdfSample,
					intersection,
					shadowIntersect,
					brdfIntersect,
					material,
					materials[shadowIntersect.materialId],
					materials[brdfIntersect.materialId],
					num_lights,
					rng);

				//pathSegments[idx].color = glm::clamp(pathSegments[idx].color, 0.f, 1.f);
				//pathSegments[idx].color = glm::vec3(shadowIntersect.t, 0.f, 0.f)/10.f;
				//pathSegments[idx].remainingBounces = 0;

#else
				scatterRay(pathSegments[idx], 
					intersection.intersect, 
					intersection.surfaceNormal, 
					material, 
					rng);
#endif

			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

// We go through the geoms and label the ones that are lights. This allows us to stream compact
// to find the light sources when we want to do direct lighting.
__global__ void findLights(int nGeom, Geom * geoms, Material * materials) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nGeom) {
		if (materials[geoms[idx].materialid].emittance > 0.0001f) {
			geoms[idx].isLight = 1;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

__global__ void findMaterialIds(int nPaths, int * ids, ShadeableIntersection * isx) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		ids[index] = isx[index].materialId;
	}
}

// Used for stream compaction in thrust::partition
struct deadPaths
{
	__host__ __device__
		bool operator()(const PathSegment& pathSeg) {
		return pathSeg.remainingBounces > 0;
	}
};

// Use this to stream compact the geoms to find the lights. This allows us to do direct lighting.
struct lights
{
	__host__ __device__
		bool operator()(const Geom& geom) {
		return geom.isLight > 0;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = BLOCKSIZE;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

#if TIME
	float total = 0.f;
	float milliseconds = 0.f;
	float intersect_time = 0.f;
	float pathtract_time = 0.f; 
	float stream_time = 0.f;
	float material_time = 0.f;
	cudaEvent_t start, end;
	printf("Iteration: %d\n", iter);
#endif

#if TIME
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
#endif
	// Generate rays depending on macro settings
#if CACHEFIRSTRAY && !ANTIALIAS
	if (firstTime) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_first_cache);
		firstTime = false;
	}
	cudaMemcpy(dev_paths, dev_first_cache, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
#else
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
#endif
	
#if TIME
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&milliseconds, start, end);
	total += milliseconds;
	printf("Generate rays: %4.4f \n", milliseconds);
#endif

	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
#if TIME
	printf("Starting number of paths: %d\n", num_paths);
#endif

	// --- Find lights for direct lighting ---
	// 0. Realized this only needs to done once if you make numLights a global var!
	// 1. Label the geoms using the findLights kernel.
	// 2. Stream compact to put all the lights in front. Then we can track which ones are lights.
#if DIRECTLIGHTING

#if TIME
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
#endif

	if (iter == 1) {
		dim3 numblocksFindLights = (hst_scene->geoms.size() + blockSize1d - 1) / blockSize1d;
		findLights << <numblocksFindLights, blockSize1d >> >(hst_scene->geoms.size(), dev_geoms, dev_materials);
		Geom* lightEnd = thrust::partition(
			thrust::device,
			dev_geoms,
			dev_geoms + hst_scene->geoms.size(),
			lights());
		numLights = lightEnd - dev_geoms;
	}
	checkCUDAError("Counting number of lights for direct lighting");

#if TIME
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&milliseconds, start, end);
	total += milliseconds;
	printf("Finding Lights: %4.4f \n", milliseconds);
#endif

#endif

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	while (!iterationComplete) {
		
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if TIME
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
#endif

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
		cudaDeviceSynchronize();

#if TIME
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&milliseconds, start, end);
		total += milliseconds;
		intersect_time += milliseconds;
		//printf("Round of intersections: %4.4f \n", milliseconds);
#endif
		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		// --- Sort by material ---
		// 1. Get IDs from the materials
		// 2. Sort paths accordingly with thrust
#if MATERIALSORT
#if TIME
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
#endif
		findMaterialIds << <numblocksPathSegmentTracing, blockSize1d >> >(
			num_paths,
			dev_materialIds,
			dev_intersections);
#if TIME
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&milliseconds, start, end);
		total += milliseconds;
		material_time += milliseconds;
		printf("Material Sort (gather mat ids): %4.4f \n", milliseconds);
#endif

#if TIME
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
#endif
		thrust::device_ptr<int> dev_thrust_keys = thrust::device_pointer_cast(dev_materialIds);
		thrust::device_ptr<PathSegment> dev_thrust_paths = thrust::device_pointer_cast(dev_paths);
		thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections = thrust::device_pointer_cast(dev_intersections);
		checkCUDAError("Allocate thrust ptr");

		thrust::stable_sort_by_key(dev_thrust_keys, dev_thrust_keys + num_paths, dev_thrust_paths);
		//thrust::stable_sort_by_key(dev_thrust_keys, dev_thrust_keys + num_paths, dev_thrust_intersections);
		//checkCUDAError("Sort materials");
#if TIME
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&milliseconds, start, end);
		total += milliseconds;
		material_time += milliseconds;
		printf("Material Sort (sorting): %4.4f \n", milliseconds);
#endif
#endif

#if TIME
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
#endif

		shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
			iter,
			depth,
			num_paths,
			numLights,
			hst_scene->geoms.size(),
			dev_geoms,
			dev_intersections,
			dev_paths,
			dev_materials
			);

#if TIME
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&milliseconds, start, end);
		total += milliseconds;
		pathtract_time += milliseconds;
		printf("Path tracing step: %4.4f \n", milliseconds);
#endif

#if STREAMCOMPACT

#if TIME
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start);
#endif
		// Stream compaction with partition
		PathSegment* new_dev_path_end = thrust::partition(
			thrust::device, 
			dev_paths, 
			dev_paths + num_paths, 
			deadPaths());
		num_paths = new_dev_path_end - dev_paths;

#if TIME
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&milliseconds, start, end);
		total += milliseconds;
		stream_time += milliseconds;
		//printf("Stream compaction: %4.4f \n", milliseconds);
		printf("Stream compactions (paths remaining): %d\n", num_paths);
#endif

#endif

		// Wait for everything to finish
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

#if STREAMCOMPACT
		iterationComplete = num_paths == 0; // TODO: should be based off stream compaction results.
#else
		iterationComplete = depth >= traceDepth || num_paths == 0;
#endif
	}

#if TIME
	printf("Total intersect: %4.4f \n", intersect_time);
	printf("Total pathtrace: %4.4f \n", pathtract_time);
	printf("Total streamcompact: %4.4f \n", stream_time);
	printf("Total materialsort: %4.4f \n", material_time);
	printf("Total: %4.4f \n", total);
#endif
	
	// Assemble this iteration and apply it to the image
	num_paths = dev_path_end - dev_paths;
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> >(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
