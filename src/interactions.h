#pragma once

#include "intersections.h"
#include <math.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
glm::vec3 normal, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = sqrt(u01(rng)); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = u01(rng) * TWO_PI;

	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}

__host__ __device__
glm::vec3 lambertBxdf(glm::vec3 in, glm::vec3 out, const Material &m){
	return m.color / PI;
}

__host__ __device__
float lambertPDF(glm::vec3 in, glm::vec3 out, glm::vec3 normal){
	return glm::dot(glm::normalize(normal), -in) / PI;
}

__host__ __device__
float lightPDF(glm::vec3 in, glm::vec3 normal, glm::vec3 intersect, glm::vec3 origin, float lightArea) {
	float cosTheta = glm::dot(-in, glm::normalize(normal));
	return glm::length(intersect - origin) / (cosTheta * lightArea);
	//return 1.f;
}

__host__ __device__
glm::vec3 lightEnergy(glm::vec3 in, glm::vec3 normal, const Material & lm) {
	return glm::dot(-in, normal) > 0 ? lm.color * lm.emittance : glm::vec3(0.f);
}

__host__ __device__
float powerHeuristic(float pdf1, float pdf2){
	return (pdf1 * pdf2) / (pdf1 * pdf1 + pdf2 * pdf2);
}

__host__ __device__
void sampleBrdf(
PathSegment & pathSeg,
glm::vec3 intersect,
glm::vec3 normal,
const Material &m,
thrust::default_random_engine &rng
){
	if (m.hasReflective > 0.f) {
		pathSeg.ray.direction = glm::reflect(pathSeg.ray.direction, normal);
	}
	else{
		pathSeg.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
	}

	pathSeg.ray.origin = intersect + (0.01f * pathSeg.ray.direction);
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
PathSegment & pathSegment,
glm::vec3 intersect,
glm::vec3 normal,
const Material &m,
thrust::default_random_engine &rng) {
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.
	if (m.hasReflective > 0.f) {
		pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
		pathSegment.color *= m.specular.color;
	}
	else{
		pathSegment.color *= m.color * glm::abs(glm::dot(normal, pathSegment.ray.direction)) / PI;
		pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
	}
	
	pathSegment.ray.origin = intersect + (0.01f * pathSegment.ray.direction);
	pathSegment.remainingBounces--;
}

__host__ __device__ 
void scatterWithDirectLighting(
	PathSegment & pathSegment,
	PathSegment & shadowFeeler,
	PathSegment & brdfSample,
	ShadeableIntersection & objIntersect,
	ShadeableIntersection & lightIntersect,
	ShadeableIntersection & brdfIntersect,
	const Material &m,
	const Material &lm,
	const Material &bm,
	int num_lights,
	thrust::default_random_engine &rng){

	// Temp color
	glm::vec3 col;

	// Direct lighting part
	glm::vec3 wi = glm::normalize(shadowFeeler.ray.direction);
	glm::vec3 lightContribution;
	glm::vec3 brdfContribution;
	float lightPdf = -0.001f;
	float brdfPdf = -0.001f;

	//// We actually hit a light with our shadow feeler! Here we calculate the lighting sample.
	//if (lightIntersect.t > 0.f && lm.emittance > 0.f) {

	//	lightContribution = lightEnergy(wi, lightIntersect.surfaceNormal, lm);
	//	lightPdf = lightPDF(wi, lightIntersect.surfaceNormal, lightIntersect.intersect, shadowFeeler.ray.origin, lightIntersect.surfaceArea);

	//	brdfContribution = lambertBxdf(wi, pathSegment.ray.direction, m);
	//	brdfPdf = lambertPDF(-wi, pathSegment.ray.direction, objIntersect.surfaceNormal);

	//	if (lightPdf > 0.f && glm::length(lightContribution) > 0.f && brdfPdf > 0.f && glm::length(brdfContribution) > 0.f) {
	//		//col[0] = lightContribution[0] * brdfPdf * brdfContribution[0];
	//		//col[1] = lightContribution[1] * brdfPdf * brdfContribution[1];
	//		//col[2] = lightContribution[2] * brdfPdf * brdfContribution[2];

	//		float dot_pdf = glm::abs(glm::dot(-wi, glm::normalize(objIntersect.surfaceNormal))) / lightPdf;
	//		float w = powerHeuristic(lightPdf, brdfPdf);
	//		w = 1.f;
	//		col[0] += w * brdfContribution[0] * lightContribution[0] * dot_pdf * num_lights;
	//		col[1] += w * brdfContribution[1] * lightContribution[1] * dot_pdf * num_lights;
	//		col[2] += w * brdfContribution[2] * lightContribution[2] * dot_pdf * num_lights;
	//	}
	//}
	//else {
	//	col = glm::vec3(0.f);
	//}

	// Reset variables
	wi = glm::vec3(0.f);
	brdfContribution = glm::vec3(0.f);
	lightContribution = glm::vec3(0.f);
	lightPdf = -0.001f;
	brdfPdf = -0.001f;

	// Compute brdf contribution
	wi = glm::normalize(brdfSample.ray.direction);
	brdfContribution = lambertBxdf(wi, pathSegment.ray.direction, m);
	brdfPdf = lambertPDF(-wi, pathSegment.ray.direction, objIntersect.surfaceNormal);
	if (brdfPdf > 0.f && glm::length(brdfContribution) > 0.f) {

		// Brdf sample hit a light
		if (brdfIntersect.t > 0.f && bm.emittance > 0.f) {
			lightContribution = lightEnergy(wi, brdfIntersect.surfaceNormal, bm);
			lightPdf = lightPDF(wi, brdfIntersect.surfaceNormal, brdfIntersect.intersect, brdfSample.ray.origin, brdfIntersect.surfaceArea);

			if (lightPdf > 0.f && glm::length(lightContribution) > 0.f) {
				float dot_pdf = glm::abs(glm::dot(wi, glm::normalize(objIntersect.surfaceNormal))) / brdfPdf;
				float w = powerHeuristic(brdfPdf, lightPdf);
				w = 1.f;
				col[0] += w * brdfContribution[0] * lightContribution[0] * dot_pdf;
				col[1] += w * brdfContribution[1] * lightContribution[1] * dot_pdf;
				col[2] += w * brdfContribution[2] * lightContribution[2] * dot_pdf;
			}
		}
	}

	//col = glm::clamp(col, 0.f, 1.f);

	// Set up new path segments
	sampleBrdf(pathSegment,
		objIntersect.intersect,
		objIntersect.surfaceNormal,
		m,
		rng);
	pathSegment.remainingBounces--;

	// Add new colors
	pathSegment.color += pathSegment.throughput * col;

	// Early exit
	if (glm::length(brdfContribution) <= 0.f || brdfPdf <= 0.f) {
		pathSegment.remainingBounces = 0;
	}

	// Update throughput
	brdfContribution *= glm::abs(glm::dot(wi, objIntersect.surfaceNormal)) / brdfPdf;
	pathSegment.throughput *= glm::max(glm::max(brdfContribution[0], brdfContribution[1]), brdfContribution[2]);

}